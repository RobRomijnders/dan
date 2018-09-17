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
from utils.util_run import setup_logger
style.use('fivethirtyeight')


def plot_embedding(ax_element, embedding, labels, label_names, colors, title):
    label_names_copy = copy(label_names)
    N = len(labels)
    for i in np.random.permutation(N):
        lbl = int(labels[i])
        ax_element.scatter(embedding[i, 0], embedding[i, 1], c=colors[lbl], marker='*',
                           s=40, label=label_names_copy[lbl])
        label_names_copy[lbl] = "_nolegend_"  # Prevent repeating legend
    ax_element.legend()
    ax_element.set_title(title)


def plot_embeddings(embeddings_and_labels,
                    experiment_names,
                    path_plotting):
    # Do some plotting
    domain_colors = ['m', 'g']

    fig, axarr = plt.subplots(2, 2, figsize=(30, 30))
    for i, ((embeddings, domain_labels, domain_label_names), experiment_name) in enumerate(zip(embeddings_and_labels, experiment_names)):
        log.debug('fit tSNE')
        model = TSNE(n_components=2, verbose=2, perplexity=100, min_grad_norm=1E-4)
        embeddings = model.fit_transform(np.copy(embeddings))

        log.debug('plot tSNE')
        plot_embedding(axarr[i, 0], embeddings, domain_labels, domain_label_names, domain_colors,
                       f'{experiment_name}: tSNE embedding, colored per domain')

        log.debug('fit PCA')
        model = PCA(n_components=2)
        embeddings = model.fit_transform(np.copy(embeddings))

        log.debug('plot PCA domain')
        plot_embedding(axarr[i, 1], embeddings, domain_labels, domain_label_names, domain_colors,
                       f'{experiment_name}: PCA embedding, colored per domain')

    for axrow in axarr:
        for ax in axrow:
            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])

    plt.suptitle(str(experiment_names))
    # plt.show()
    fig.savefig(join(path_plotting, 'rep_retrieval.pdf'), dpi=1500, format='pdf', transparent=True)


def read_embeddings(experiments, subsample=None):
    for experiment in experiments:
        # Read the embeddings
        embedding_filepaths = glob.glob(join(experiment + '_*__embeddings.csv'))
        assert len(embedding_filepaths) > 0, "Found no embeddings of experiment %s in path %s" % (experiment_name, path)

        embeddings = []
        domain_labels = []
        domain_label_names = []
        segm_labels = []
        decisions = []

        for i, filepath in enumerate(sorted(embedding_filepaths)):
            data = np.loadtxt(filepath, delimiter=',')
            embedding = data[:, :-1]
            decision = data[:, -1]
            num_samples = embedding.shape[0]

            embeddings.append(embedding)
            domain_labels.append(i*np.ones((num_samples,)))
            decisions.append(decision)

            if os.path.exists(filepath.replace('embeddings', 'labels')):
                segm_labels.append(np.loadtxt(filepath.replace('embeddings', 'labels')))

            label_name, _ = parse_filepath(filepath)
            assert len(label_name) > 0
            domain_label_names.append(label_name)

        embeddings = np.concatenate(embeddings, 0)
        domain_labels = np.concatenate(domain_labels, axis=0)
        assert len(domain_labels) == embeddings.shape[0]
        if subsample:
            yield (*subsample_embeddings(embeddings, domain_labels, subsample), domain_label_names)
        else:
            yield embeddings, domain_labels, domain_label_names


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


# def find_all_experiments(path):
#     all_experiments = set()
#     for name in os.listdir(path):
#         match = re.compile('[^_]+_(gta5|cityscapes)__(labels|embeddings|bn).csv', flags=re.I).match(name)
#         if match:
#             all_experiments.add(match.group(0).split('_')[0])
#     return all_experiments


def subsample_embeddings(embeddings, domain_labels, num_samples):
    # Subsample embeddings

    num_actual_samples = embeddings.shape[0]
    assert domain_labels.shape[0] == num_actual_samples
    if num_actual_samples <= num_samples:
        ind = list(range(num_actual_samples))
    else:
        ind = np.random.choice(num_actual_samples, size=(num_samples,), replace=False)
    return embeddings[ind], domain_labels[ind]


def main(direc):
    path_plotting = '/hdd/dropbox/Dropbox/grad/results/bn_repr/scatter'
    maybe_makedirs(path_plotting)
    log.debug('Save all plots to %s' % path_plotting)

    experiments = ['/hdd/logs_overnight_training/newer/overnight_0403/fullgta5-ema/extract/fullgta5-ema',
                   '/hdd/logs_overnight_training/newer/overnight_0403/fulluda-custombatch-lambda-0.005-ema/extract/fulluda-custombatch-lambda-0.005-ema']

    embeddings_and_labels = read_embeddings(experiments, subsample=3000)


    plot_embeddings(embeddings_and_labels,
                    experiment_names=['GTA5-single', 'UADA'],
                    path_plotting=path_plotting)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize representations')
    parser.add_argument('direc', type=str,
                        help='the directory name')
    args = parser.parse_args()

    log = setup_logger(args.direc, 'plot')
    main(direc=args.direc)
