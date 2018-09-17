import numpy as np
import sys
sys.path.append('/home/mps/Documents/semantic-segmentation-fork/semantic-segmentation')
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import os
import glob
from os.path import join
from utils.utils import maybe_makedirs
from utils.util_plot import im2lbl_path, palette_lid, palette_cid, class_names


def get_im_code(fnames_list):
    for fname in fnames_list:
        yield os.path.basename(fname).replace('_leftImg8bit.png', '').replace('.png', ''), fname


dims = (768, 384)
col_names = ['Trained on only \n synthetic data (GTA5)', 'Adapted to real domain \n (Cityscapes)']

zero_image = 255*np.ones((200, 400, 3), dtype=np.uint8)

mercedes_mask = np.array(Image.open('/hdd/dropbox/Dropbox/grad/datasets/cityscapes/mask/mercedes_mask_outer.png').resize(dims)).astype(np.int32)

from matplotlib.lines import Line2D
custom_handles = [Line2D([0], [0], color=rgb_vec, lw=4, label=class_name) for (rgb_vec, class_name) in zip(palette_cid, class_names)]


def main(basedir):
    out_dir = join(basedir, 'movie')
    maybe_makedirs(out_dir, force_dir=True)

    predict_dirs = sorted([x[0] for x in os.walk(basedir) if 'predictions' in x[0]])
    num_preds = len(predict_dirs)

    image_fnames = glob.glob(join(basedir, '*.png'))
    image_codedict = {code: filename for code, filename in get_im_code(image_fnames)}

    num_fnames = len(image_fnames)

    predictions_codedicts = [{code: filename for code, filename in get_im_code(glob.glob(join(predict_dir, '*.png')))}
                             for predict_dir in predict_dirs]

    all_fnames = []
    for code, im_filename in image_codedict.items():
        if all((code in codedict for codedict in predictions_codedicts)):
            all_fnames.append([code, im_filename] + [codedict[code] for codedict in predictions_codedicts])

    for num_row, fname_row in enumerate(all_fnames):
        frame_num = fname_row[0][-5:]
        if not frame_num.isdigit():
            frame_num = 'UNK'
        if 'stuttgart_00' in basedir and int(frame_num) > 400:
            continue

        plt.figure()
        f, axarr = plt.subplots(2, 2, figsize=(12, 8))
        f.subplots_adjust(wspace=0, hspace=0)
        axarr[0, 1].imshow(Image.open(fname_row[1]).resize(dims))
        axarr[0, 1].set_title('Input image')
        axarr[0, 1].legend(handles=custom_handles, loc=7, bbox_to_anchor=(0.9, -1.43), ncol=10, fontsize='small', markerscale=5)


        base_font = 14
        axarr[0, 0].imshow(zero_image)
        axarr[0, 0].text(10, 50, 'Domain Agnostic Normalization Layer', fontdict={'family': 'serif', 'size': base_font})
        axarr[0, 0].text(10, 70, 'for Unsupervised Adversarial Domain adaptation', fontdict={'family': 'serif', 'size': base_font})
        if False:
            axarr[0,0].text(10, 110, 'R. Romijnders, P. Meletis, G. Dubbelman', fontdict={'family': 'serif', 'size': int(0.8*base_font)})
            axarr[0, 0].text(10, 130, 'Mobile Perception Systems (SPS-VCA)', fontdict={'family': 'serif', 'size': int(0.8*base_font)})
            axarr[0, 0].text(10, 150, 'TU/e Eindhoven', fontdict={'family': 'serif', 'size': int(0.8*base_font)})
        else:
            axarr[0, 0].text(10, 110, 'Info removed for blind review', fontdict={'family': 'serif', 'size': int(0.8 * base_font)})
        axarr[0, 0].text(10, 170, 'June, 2018', fontdict={'family': 'serif', 'size': int(0.8*base_font)})
        axarr[0, 0].text(300, 190, f'frame {frame_num}', fontdict={'family': 'serif', 'size': int(0.7*base_font)})

        for num_pred in range(num_preds):
            label = np.array(Image.open(fname_row[num_pred+2]).resize(dims))
            # Mask out the Mercedes
            label = np.clip(label + mercedes_mask, 0, 20)
            label_array = np.take(palette_cid, label, axis=0)
            axarr[1, num_pred].imshow(label_array)

        for n_row, axrow in enumerate(axarr):
            for n_col, ax in enumerate(axrow):
                ax.get_xaxis().set_ticklabels([])
                ax.get_yaxis().set_ticklabels([])
                if n_row == 1:
                    ax.set_title(col_names[n_col])

                ax.tick_params(
                    which='both',  # both major and minor ticks are affected
                    bottom='off',  # ticks along the bottom edge are off
                    top='off',  # ticks along the top edge are off
                    left='off',
                    right='off',
                    labelbottom='off')  # labels along the bottom edge are off
        plt.tight_layout()
        f.savefig(join(out_dir, f'{fname_row[0]}.png'), format='png')
        plt.close('all')
        print(f'{num_row:5.0f} out of {num_fnames:6.0f}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='base_directory')
    parser.add_argument('direc', type=str,
                        help='the directory name')
    args = parser.parse_args()
    main(args.direc)