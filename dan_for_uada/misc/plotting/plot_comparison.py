from glob import glob
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from os.path import basename, join
from utils.util_plot import im2lbl_path, palette_lid, palette_cid
from utils.utils import maybe_makedirs
from input.make_tfrecord import get_apollo_im_label_fnames
import json


def plot_comparison_table(filenames, plot_keys, coloring_types, out_dir, dims=(768, 384)):
    """

    :param filenames: dict of filenames. Key is the label type and value is a list of path to labels
    :param plot_keys: which keys of the dict filenames to plot
    :param coloring_types: for each plot_key this should mention what coloring type is used
    :param out_dir: output dir to save the comparison figure. if empty string, then display interactively
    :param dims: dimensions to resize all images and labels too. Note that in format (WIDTH, HEIGHT)
    :return:
    """
    # Some basic assertions to start with
    assert 'Image' in filenames, f'Your filenames dict must at least contain Image'
    for color_type in coloring_types:
        assert color_type in ['lid', 'cid', 'lid_map', 'lid_apo']
    assert len(filenames['Image']) > 0
    assert len(plot_keys) == len(coloring_types)
    if dims[0] < dims[1]:
        print(f'WARNING: the dims are interpreted as width={dims[0]} and height={dims[1]}')
    assert out_dir != ''
    out_dir_pred = join(out_dir, 'preds/')
    maybe_makedirs(out_dir_pred, force_dir=True)
    print(f'we have filenames for keys {filenames.keys()} and we are plotting {plot_keys}')

    # Set up constants for plotting
    num_col = len(plot_keys) + 1 # +1 for the images column
    num_row = 5

    # Make a pyplot figure
    f, axarr = plt.subplots(num_row, num_col)

    count_row = 0  # Count in which row we are plotting
    for filename in filenames['Image']:
        # Code is structured as follows:
        # -1 Loop over image filenames
        # -2 extract the image code
        # -3 make sure that all other keys also have an image with that code
        # -4 only if all other filenames are found, then plot the row
        im_code = basename(filename).replace('_leftImg8bit.png', '').replace('.jpg', '')

        def find_other_filenames():
            """
            Find all the filenames whose code appears in all plot_keys
            :return:
            """
            comparison_fnames = []
            for key in plot_keys:
                matching_fnames = list(filter(lambda filepath: im_code in filepath, filenames[key]))

                if len(matching_fnames) == 0:
                    print(f'Image code {im_code} not found in list with key {key}')
                    return None
                elif len(matching_fnames) == 1:
                    comparison_fnames.append(matching_fnames[0])
                else:
                    assert False, f'Found multiple hits on code {im_code}'
            return comparison_fnames

        comparison_files = find_other_filenames()
        if comparison_files is None:
            # Apparently, the image code was not found in all methods
            continue
        else:
            count_row += 1  # Only increment the count_row if we actually found a match in all plot_keys
        if count_row > num_row:
            break

        input_image = Image.open(filename).resize(dims)
        axarr[count_row - 1, 0].imshow(input_image)
        input_image.save(join(out_dir_pred, f'{count_row}_image.png'))
        del input_image

        for n_col, (fname_comparison, color_type) in enumerate(zip(comparison_files, coloring_types)):
            label = Image.open(fname_comparison).resize(dims)

            # Convert the labels according to their color type and palette
            if color_type == 'lid':
                label_array = np.take(palette_lid, np.array(label), axis=0)
            elif color_type == 'lid_map':
                lids2cids_map = [-1, -1, -1, 4, -1, -1, 3, -1, -1, -1, -1, -1, -1, 0, -1, 1, -1, 2, -1,11,12,12,12, -1, -1, -1, -1,10, -1, 9, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  5, -1,  5,  6, -1,  7, -1,  18, -1, 15, 13, -1, 17, 16, -1, -1, 14, -1, -1, -1, -1]
                label_cid = np.take(lids2cids_map, np.array(label), axis=0)
                label_array = np.take(palette_cid, label_cid, axis=0)
            elif color_type == 'cid':
                label_array = np.take(palette_cid, np.array(label), axis=0)
            elif color_type == 'lid_apo':
                with open('/home/mps/Documents/rob/datasets/problem_uda_apollo.json') as fp:
                    lids2cids_apo = json.load(fp)['lids2cids']
                label_cid = np.take(lids2cids_apo, np.array(label), axis=0)
                label_array = np.take(palette_cid, label_cid, axis=0)
            else:
                assert False, 'Color type not recognised'

            axarr[count_row - 1, n_col + 1].imshow(label_array)
            Image.fromarray((label_array * 255).astype(np.uint8)).save(join(out_dir_pred, f'{count_row}_{n_col}.png'))

    # All the pyplot magic :)
    col_names = ['Image'] + plot_keys
    for n_row, axrow in enumerate(axarr):
        for n_col, ax in enumerate(axrow):
            if n_row == 0:
                ax.set_title(col_names[n_col])

            ax.get_xaxis().set_ticklabels([])
            ax.get_yaxis().set_ticklabels([])

            ax.tick_params(
                which='both',  # both major and minor ticks are affected
                bottom='off',  # ticks along the bottom edge are off
                top='off',  # ticks along the top edge are off
                left='off',
                right='off',
                labelbottom='off')  # labels along the bottom edge are off
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    # Either show image or save it
    if out_dir is '':
        plt.show()
    else:
        print(f'Start saving to {out_dir}')
        f.savefig(join(out_dir, 'comparison.pdf'), dpi=1000, format='pdf', transparent=True)


if __name__ == '__main__':
    dataset = 'mapillary'

    # filenames is a dict with
    #  * keys: the name of the content, for example Image, Label or the Model name that was used for prediction
    #  * values: a list of filenames corresponding to the key. Some code will later figure out the intersection of
    #            the filenames that occurs across the keys
    filenames = dict()
    if dataset == 'cityscapes':
        filenames['Image'] = glob('/hdd/datasets/cityscapes/leftImg8bit/val/frankfurt/*.png')
        filenames['Label'] = list(map(lambda x: im2lbl_path(x, 'cityscapes'), filenames['Image']))
        filenames['Source-only'] = glob('/hdd/logs_overnight_training/newer/overnight_0403/fullgta5-ema/predictions/*.png')
        filenames['UADA'] = glob('/hdd/logs_overnight_training/newer/overnight_0403/fulluda-custombatch-lambda-0.005-ema/predictions/*.png')
        filenames['AdaBN'] = glob('/hdd/logs_overnight_training/newer/overnight_0403/fullgta5-ema-ADAP-cityscapes-0500/predictions/*.png')

        out_dir = '/hdd/dropbox/Dropbox/grad/results/comparison_figures/cityscapes/'

        plot_comparison_table(filenames, ['Label', 'Source-only', 'UADA'], ['lid', 'lid', 'lid'], out_dir=out_dir)
    elif dataset == 'mapillary':
        filenames['Image'] = glob('/hdd/datasets/mapillary/validation/images/*.jpg')
        filenames['Label'] = list(map(lambda x: im2lbl_path(x, 'mapillary'), filenames['Image']))
        filenames['Source-only'] = glob('/hdd/logs_overnight_training/newer/overnight_0403/fullgta5-ema/predictions_mapillary/*.png')
        filenames['UADA'] = glob(
            '/hdd/logs_overnight_training/newer/overnight_0403/fulluda-custombatch-lambda-0.005-ema/predictions_mapillary/*.png')

        out_dir = '/hdd/dropbox/Dropbox/grad/results/comparison_figures/mapillary_unseen'
        maybe_makedirs(out_dir, force_dir=True)

        plot_comparison_table(filenames, ['Label', 'Source-only', 'UADA'], ['lid_map', 'cid', 'cid'], out_dir=out_dir)

    elif dataset == 'apollo':
        import random
        random.seed(124)
        image_fnames, label_fnames = get_apollo_im_label_fnames('/hdd/datasets/apolloscape/original')
        filenames['Image'] = random.sample(image_fnames, 200)
        filenames['Label'] = label_fnames
        filenames['Source-only'] = glob('/hdd/logs_overnight_training/newer/overnight_0509/fullgta-1/predict_apollo/*.png')
        filenames['UADA'] = glob('/hdd/logs_overnight_training/newer/overnight_0509/fulluda-1/predict_apollo/*.png')

        out_dir = '/hdd/dropbox/Dropbox/grad/results/comparison_figures/apollo_unseen/'
        maybe_makedirs(out_dir, force_dir=True)

        plot_comparison_table(filenames,
                              ['Label', 'Source-only', 'UADA'],
                              ['lid_apo', 'cid', 'cid'],
                              out_dir=out_dir)

