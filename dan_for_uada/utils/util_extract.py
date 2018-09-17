import numpy as np
import os
from os.path import join
from scipy import misc
from skimage.transform import resize
from PIL import Image
from math import floor
from itertools import groupby
import matplotlib.pyplot as plt


palette_class = [[128, 64, 128], [244, 35, 232], [70, 70, 70],   [102, 102, 156], [190, 153, 153],
                 [153, 153, 153],[250, 170, 30], [220, 220, 0],  [107, 142, 35], [152, 251, 152],
                 [70, 130, 180], [220, 20, 60],  [255, 0, 0],    [0, 0, 142],    [0, 0, 70],
                 [0, 60, 100],   [0, 80, 100],   [0, 0, 230],    [119, 11, 32],
                 [255, 0, 0], [0, 0, 0]]
palette_class = np.array(palette_class)/255.


def make_palette_label():

    def extend_palette():
        for lid in [-1, -1, -1, -1, -1, -1, -1,  0,  1, -1, -1,  2,  3,  4, -1, -1, -1,  5, -1,  6,
                    7,  8,  9, 10, 11, 12, 13, 14, 15, -1, -1, 16, 17, 18, -1]:
            lid_ = lid if lid is not -1 else 19
            yield palette_class[lid_]

    return np.array(list(extend_palette()))


def plot_im_lbl_dec(images, labels, decisions, experiment='', dataset='', output_dir='/tmp', suffix=''):
    num_samples = len(images)
    assert len(labels) == len(decisions) == num_samples

    fig, axarr = plt.subplots(num_samples, 3)
    palette_label = make_palette_label()

    for num, (image, label, decision) in enumerate(zip(images, labels, decisions)):
        # Plot the image
        axarr[num, 0].imshow(image)
        axarr[num, 0].get_xaxis().set_ticklabels([])
        axarr[num, 0].get_yaxis().set_ticklabels([])

        # Plot the labels
        label_array = np.take(palette_label, label, axis=0)
        axarr[num, 1].imshow(label_array)
        axarr[num, 1].get_xaxis().set_ticklabels([])
        axarr[num, 1].get_yaxis().set_ticklabels([])

        # Plot the decisions
        decision_array = np.take(palette_class, decision, axis=0)
        axarr[num, 2].imshow(decision_array)
        axarr[num, 2].get_xaxis().set_ticklabels([])
        axarr[num, 2].get_yaxis().set_ticklabels([])

    # PyPlot Magic
    for axrow in axarr:
        for ax in axrow:
            ax.tick_params(
                which='both',  # both major and minor ticks are affected
                bottom='off',  # ticks along the bottom edge are off
                top='off',  # ticks along the top edge are off
                left='off',
                right='off',
                labelbottom='off')  # labels along the bottom edge are off

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    plt.suptitle(f'Experiment {experiment}, evaluated on {dataset}')
    print("start saving")
    # plt.show()
    fig.savefig(join(output_dir, f'samples_{experiment}_{dataset}{suffix}.pdf'), dpi=1000,
                format='pdf', transparent=True)


def yield_image_label_decision(output, dataset):
    probs = output['probabilities']
    assert len(probs.shape) == 3
    decisions = np.argmax(probs, axis=2)

    labels = get_labels_from_path(output['rawimagespaths'], dataset)

    raw_image = output['rawimages']
    assert len(raw_image.shape) == 3
    assert raw_image.shape[2] == 3  # For RGB image
    yield raw_image, np.array(labels), decisions


def get_domain_logit(output, dataset, lids2cids, domain_label):
    assert domain_label in [0, 1]

    domain_logits = output['domain_logits']

    H_rep, W_rep = domain_logits.shape

    labels, label_stride = get_labels(output['rawimagespaths'], dataset, H_rep, W_rep)

    for x in range(H_rep):
        for y in range(W_rep):
            label_majority = np.argmax(np.bincount(
                labels[label_stride * x:label_stride * (x + 1),
                label_stride * y:label_stride * (y + 1)].flatten()))

            cid = lids2cids[label_majority]

            domain_correct = int(domain_logits[x, y] > 0.) == domain_label
            yield cid, domain_correct





def output_to_representations(*args, **kwargs):
    def yield_representations(output, lids2cids, dataset, random_sampling=True):
        reps = output['representations']
        probs = output['probabilities']

        N = 10  # Number of representations to subsample

        H_rep, W_rep, C_rep = reps.shape
        H_prob, W_prob, C_prob = probs.shape

        assert H_prob % H_rep == 0
        stride = int(round(H_prob / H_rep))

        labels, label_stride = get_labels(output['rawimagespaths'], dataset, H_rep, W_rep)

        if random_sampling:
            x_sample = np.random.randint(0, H_rep, size=(N, ), dtype=np.int32)
            y_sample = np.random.randint(0, W_rep, size=(N, ), dtype=np.int32)

            for i, (x, y) in enumerate(zip(x_sample, y_sample)):
                rep = reps[x, y]

                prob = np.mean(probs[stride*x:stride*(x+1), stride*y:stride*(y+1)], axis=(0, 1))
                label_majority = np.argmax(np.bincount(
                    labels[label_stride*x:label_stride*(x+1), label_stride*y:label_stride*(y+1)].flatten()))

                decision = np.argmax(prob)
                cid = lids2cids[label_majority]

                yield rep, decision, cid
        else:
            for x in range(H_rep):
                for y in range(W_rep):
                    rep = reps[x, y]

                    prob = np.mean(probs[stride * x:stride * (x + 1), stride * y:stride * (y + 1)], axis=(0, 1))
                    label_majority = np.argmax(np.bincount(
                        labels[label_stride * x:label_stride * (x + 1),
                               label_stride * y:label_stride * (y + 1)].flatten()))

                    decision = np.argmax(prob)
                    cid = lids2cids[label_majority]

                    yield rep, decision, cid

    if kwargs.get('random_sampling', True):
        yield from yield_representations(*args, **kwargs)
    else:
        # Average the representations per class
        all_output = sorted(yield_representations(*args, **kwargs), key=lambda tup: tup[2])
        for key, group in groupby(all_output, key=lambda tup: tup[2]):
            reps, decisions, labels = list(zip(*group))
            reps_mean = np.mean(np.stack(reps), axis=0)
            yield reps_mean, decisions[0], key


def get_labels_from_path(im_path, dataset):
    if dataset == 'cityscapes':
        label_path = im_path.decode("utf-8").replace('leftImg8bit.png', 'gtFine_labelIds.png').replace('leftImg8bit/',
                                                                                                       'gtFine/')
        if not os.path.exists(label_path):
            print('Labelpath not found (%s)' % label_path)
            return None, None
        labels = Image.open(label_path)

    elif dataset == 'gta5':
        label_path = im_path.decode("utf-8").replace('images', 'labels')

        if not os.path.exists(label_path):
            print('Labelpath not found (%s)' % label_path)
            return None, None

        labels = Image.open(label_path)
    else:
        print(f'No such dataset implemented {dataset}')
        labels = None
    return labels


def get_labels(im_path, dataset, H, W):
    labels = get_labels_from_path(im_path, dataset)
    W_lab, H_lab = labels.size
    if not (H_lab % H == 0 and W_lab % W == 0):
        largest_ratio = min(floor(H_lab / H), floor(W_lab / W))
        new_shape = (W*largest_ratio, H*largest_ratio)  # Be careful, H and W are switched
        print(f'Resize image from {labels.size} to {new_shape}')
        labels = labels.resize(new_shape, resample=Image.NEAREST)
    labels = np.array(labels)

    assert labels.shape[0] < labels.shape[1]
    assert labels.shape[0] % H == 0, \
        "stride is not an integer along height for %i in representations and %i in labels" % \
        (H, labels.shape[0])
    assert labels.shape[1] % W == 0
    label_stride = int(labels.shape[0] / H)
    assert label_stride == int(labels.shape[1] / W)

    return labels, label_stride
