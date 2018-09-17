import numpy as np
import glob
from os.path import join
import os
import re
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.style as style
from copy import copy
import json
from utils.utils import maybe_makedirs
import argparse
import logging
from scipy.spatial.distance import pdist, squareform
from utils.util_run import setup_logger
style.use('fivethirtyeight')


# def plot_embeddings(features, decisions, domain_labels, segm_labels, domain_label_names,
#                     segm_label_names, segm_colors, path_plotting='/tmp', acc="", experiment_name="UNKNOWN"):
#     assert domain_labels.shape[0] == segm_labels.shape[0] == decisions.shape[0], \
#         "sizes do not match"
#     assert len(np.unique(domain_labels)) <= len(domain_label_names), \
#         "Found unique domains than domain names"
#     assert len(np.unique(segm_labels)) <= len(segm_label_names), \
#         "Found more unique segmentation labels than defined segmentation label names"
#     assert len(np.unique(decisions)) <= len(segm_label_names), \
#         "Found more unique segmentation decisions than segmentation label names"
#     assert len(segm_label_names) <= len(segm_colors), "Found more segmentation labels than colors"
#     # Do some plotting
#     domain_colors = ['g', 'm']
#     all_markers = ['*', '+', 'o', 'v', '>', '<', '^', '1', '2', '3', '4', 'x', 'd', 'h', '|', '_', 's']*5
#
#     fig, axarr = plt.subplots(3, 2, figsize=(30, 20))
#
#     def plot_embedding(ax_element, embedding, labels, label_names, colors, title):
#         label_names_copy = copy(label_names)
#         for i, lbl in enumerate(labels):
#             lbl = int(lbl)
#             ax_element.scatter(embedding[i, 0], embedding[i, 1], c=colors[lbl], marker=all_markers[lbl],
#                                s=50, label=label_names_copy[lbl])
#             label_names_copy[lbl] = "_nolegend_"  # Prevent repeating legend
#         if 'decision' not in title:
#             ax_element.legend()
#         ax_element.set_title(title)
#
#     # log.debug('fit tSNE')
#     # model = TSNE(n_components=2, verbose=2, perplexity=100, min_grad_norm=1E-4)
#     # embeddings = model.fit_transform(np.copy(features))
#     #
#     # log.debug('plot tSNE')
#     # plot_embedding(axarr[0, 0], embeddings, domain_labels, domain_label_names, domain_colors,
#     #                'tSNE embedding, colored per domain')
#     # plot_embedding(axarr[1, 0], embeddings, segm_labels, segm_label_names, segm_colors,
#     #                'tSNE embedding, colored per class')
#     # plot_embedding(axarr[2, 0], embeddings, decisions, segm_label_names, segm_colors,
#     #                'tSNE embedding, colored per decision')
#
#     log.debug('fit PCA')
#     model = PCA(n_components=2)
#     embeddings = model.fit_transform(np.copy(features))
#
#     log.debug('plot PCA domain')
#     plot_embedding(axarr[0, 1], embeddings, domain_labels, domain_label_names, domain_colors,
#                    'PCA embedding, colored per domain')
#     plot_embedding(axarr[1, 1], embeddings, segm_labels, segm_label_names, segm_colors,
#                    'PCA embedding, colored per class')
#     plot_embedding(axarr[2, 1], embeddings, decisions, segm_label_names, segm_colors,
#                    'PCA embedding, colored per decision')
#
#     for axrow in axarr:
#         for ax in axrow:
#             ax.get_xaxis().set_ticklabels([])
#             ax.get_yaxis().set_ticklabels([])
#
#     # plt.show()
#     fig.savefig(join(path_plotting, 'repr_%s_%s.pdf' %
#                 (experiment_name, acc)), dpi=1500, format='pdf', transparent=True)


def read_embeddings(path, experiment_name):
    # Read the embeddings
    embedding_filepaths = glob.glob(join(path, experiment_name + '_*__embeddings.csv'))
    assert len(embedding_filepaths) > 0, "Found no embeddings of experiment %s in path %s" % (experiment_name, path)

    embeddings = []
    decisions = []
    domain_labels = []
    domain_label_names = []
    segm_labels = []

    for i, filepath in enumerate(sorted(embedding_filepaths)):
        data = np.loadtxt(filepath, delimiter=',')
        if os.path.exists(filepath.replace('embeddings', 'labels')):
            segm_labels.append(np.loadtxt(filepath.replace('embeddings', 'labels')))

        embedding = data[:, :-1]
        decision = data[:, -1]
        num_samples = embedding.shape[0]

        embeddings.append(embedding)
        decisions.append(decision)
        domain_labels.append(i*np.ones((num_samples,)))

        label_name, _ = parse_filepath(filepath)
        domain_label_names.append(label_name)
        print(f'dataset {label_name} has index {i}')

    embeddings = np.concatenate(embeddings, 0)
    domain_labels = np.concatenate(domain_labels, axis=0)
    segm_labels = np.concatenate(segm_labels, axis=0)
    decisions = np.concatenate(decisions, axis=0)
    assert len(segm_labels) == len(decisions) == len(domain_labels) == embeddings.shape[0]
    return embeddings, decisions, domain_labels, domain_label_names, segm_labels


def parse_filepath(filepath):
    try:
        label_name = re.compile("(_[^_]+__)").search(filepath).group(0)[1:-2]
    except AttributeError:
        label_name = 'UNKNOWN_LABEL'

    try:
        experiment_name = re.compile("([^_]+_)").search(os.path.basename(filepath)).group(0)[:-1]
    except AttributeError:
        experiment_name = 'UNKNOWN_EXPERIMENT'
    return label_name, experiment_name


def subsample_embeddings(embeddings, domain_labels, segm_labels, num_samples):
    # Subsample embeddings

    num_actual_samples = embeddings.shape[0]
    assert domain_labels.shape[0] == segm_labels.shape[0] == num_actual_samples
    if num_actual_samples <= num_samples:
        ind = list(range(num_actual_samples))
    else:
        ind = np.random.choice(num_actual_samples, size=(num_samples,), replace=False)
    return embeddings[ind], domain_labels[ind], segm_labels[ind]


def train_knn(data, labels):
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels, test_size=0.2)

    model = KNeighborsClassifier().fit(data_train, labels_train)
    acc = model.score(data_test, labels_test)

    log.debug('kNN with 5 nearest neighbors achieves %5.3f accuracy on test set' % acc)
    return acc


def retrieve_plot(data, labels, label_names):
    D = squareform(pdist(data))

    N = len(labels)
    assert N % 2 == 0
    assert np.sum(labels[:int(N/2)]) == 0

    def get_retrievals():
        for num_neigh in range(1, 50):
            retrievals = []
            for n in range(int(N/2)):
                dist = D[n]
                indices_closest = np.argsort(dist)[1:num_neigh+1]
                assert len(indices_closest.shape) == 1
                assert len(indices_closest) == num_neigh
                class_closest = labels[indices_closest]
                retrievals.append(np.mean(class_closest == 0))
            ave_retrieval = np.mean(retrievals)
            print(f'at {num_neigh:8.0f}, the retrieval is {ave_retrieval:10.8f}')
            yield ave_retrieval
    retrieval_per_num_neigh = list(get_retrievals())
    return retrieval_per_num_neigh


def make_color_tuples(segm_colors):
    """
    prepares the colors specs for PyPlot formatting
    :param segm_colors:
    :return:
    """
    for rgb_list in segm_colors:
        yield tuple((value/255. for value in rgb_list))


def parse_problem_def(path):
    with open(path) as f:
        problem_def = json.load(f)
    segm_colors = problem_def['cids2colors']
    segm_label_names = problem_def['cids2labels']

    segm_colors = list(make_color_tuples(segm_colors))

    # Do +1 in next line as the licenseplate class is not included in names list
    assert len(segm_colors) == len(segm_label_names) + 1
    return segm_label_names, segm_colors


def find_all_experiments(path):
    all_experiments = set()
    for name in os.listdir(path):
        match = re.compile('[^_]+_(gta5|cityscapes)__(labels|embeddings).csv', flags=re.I).match(name)
        if match:
            all_experiments.add(match.group(0).split('_')[0])
    return all_experiments


def main(direc):
    path = direc
    path_to_problem_def = '/home/mps/Documents/rob/training22/problem_gta5_19.json'
    path_plotting = join(path, 'plots/')
    maybe_makedirs(path_plotting)
    log.debug('Save all plots to %s' % path_plotting)

    segm_label_names, segm_colors = parse_problem_def(path_to_problem_def)

    all_experiments = find_all_experiments(path)
    for i, experiment_name in enumerate(all_experiments):
        log.debug('\n\nExperiment %15s %5i/%5i' % (experiment_name, i, len(all_experiments)))
        embeddings, decisions, domain_labels, domain_label_names, segm_labels = \
            read_embeddings(path, experiment_name)

        acc = train_knn(embeddings, domain_labels)
        print(acc)

        retrievals = retrieve_plot(embeddings, domain_labels, domain_label_names)
        np.savetxt(fname=join(path, 'retrievals.csv'), X=retrievals, delimiter=',')


        #
        # plot_embeddings(embeddings, decisions,
        #                 domain_labels, segm_labels,
        #                 domain_label_names, segm_label_names,
        #                 segm_colors, acc="%.3f" % acc,
        #                 experiment_name=experiment_name,
        #                 path_plotting=path_plotting)
        #

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize representations')
    parser.add_argument('direc', type=str,
                        help='the directory name')
    args = parser.parse_args()

    log = setup_logger(args.direc, 'plot')
    main(direc=args.direc)
