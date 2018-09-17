import tensorflow as tf
import numpy as np
import glob
from scipy import misc
from scipy.io import loadmat
from os.path import join
from PIL import Image
import time
from utils.utils import maybe_makedirs

def int64_feature(values):
    if not isinstance(values, (tuple, list)):
        values = [values]
    else:
        values = values.flatten().tolist()
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def _string_to_bytes(string):
    return tf.train.BytesList(value=[string])


def string_feature(string):
    return tf.train.Feature(bytes_list=_string_to_bytes(string))


def get_shape(fname):
    im = Image.open(fname)
    return im.size


def create_tfexample(image_string, label_string, packet, num_sample):
    feature_dict = {
        'image/encoded': string_feature(image_string),
        'image/format': string_feature(packet['format'].encode('utf-8')),
        'image/path': string_feature(packet['image_fnames'][num_sample].encode('utf-8')),
        'label/encoded': string_feature(label_string),
        'label/format': string_feature(packet['format'].encode('utf-8')),
        'label/path': string_feature(packet['label_fnames'][num_sample].encode('utf-8')),
        'height': int64_feature(packet['dims'][0]),
        'width': int64_feature(packet['dims'][1]),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature_dict))


def add_to_tfrecords(packet, tfrecord_writer, reshape):
    assert reshape in ['resize', 'crop', 'none']
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        raw_input_shapes = tf.placeholder(dtype=tf.int32, shape=[4, ])
        image_placeholder = tf.placeholder(dtype=tf.uint8)
        label_placeholder = tf.placeholder(dtype=tf.uint8)

        if reshape == 'resize':
            # Indicates the GTA5 dataset
            # For the GTA5 dataset, we must reshape the images and labels
            # For 5000 images, the label maps have different size
            # For 60 images, they are different resolution
            image_with_shape = tf.reshape(image_placeholder,
                                          shape=tf.convert_to_tensor([raw_input_shapes[0], raw_input_shapes[1], 3]))
            resized_image = tf.image.resize_nearest_neighbor(tf.expand_dims(image_with_shape, 0), list(packet['dims']))
            encoded_image = tf.image.encode_png(tf.squeeze(resized_image, 0))

            label_with_shape = tf.reshape(label_placeholder,
                                          shape=tf.convert_to_tensor([raw_input_shapes[2], raw_input_shapes[3], 1]))
            resized_label = tf.image.resize_nearest_neighbor(tf.expand_dims(label_with_shape, 0), list(packet['dims']))
            encoded_label = tf.image.encode_png(tf.squeeze(resized_label, 0))
        elif reshape == 'crop':
            tf.assert_equal(raw_input_shapes[:2], raw_input_shapes[2:])
            aspect_ratio = raw_input_shapes[1] / raw_input_shapes[0]

            new_shape = tf.cond(tf.greater_equal(aspect_ratio, 2.0),
                                true_fn=lambda: (raw_input_shapes[0], raw_input_shapes[0]*2),
                                false_fn=lambda: (tf.floor_div(raw_input_shapes[1], 2), raw_input_shapes[1]))

            image_with_shape = tf.reshape(image_placeholder,
                                          tf.convert_to_tensor([raw_input_shapes[0], raw_input_shapes[1], 3]))
            cropped_image = tf.image.resize_image_with_crop_or_pad(image_with_shape, new_shape[0], new_shape[1])
            resized_image = tf.image.resize_nearest_neighbor(tf.expand_dims(cropped_image, 0), list(packet['dims']))
            encoded_image = tf.image.encode_png(tf.squeeze(resized_image, 0))

            label_with_shape = tf.reshape(label_placeholder,
                                          shape=tf.convert_to_tensor([raw_input_shapes[2], raw_input_shapes[3], 1]))
            cropped_label = tf.image.resize_image_with_crop_or_pad(label_with_shape, new_shape[0], new_shape[1])
            resized_label = tf.image.resize_nearest_neighbor(tf.expand_dims(cropped_label, 0), list(packet['dims']))
            encoded_label = tf.image.encode_png(tf.squeeze(resized_label, 0))

        elif reshape == 'none':
            image_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_image = tf.image.encode_png(image_placeholder)
            label_placeholder = tf.placeholder(dtype=tf.uint8)
            encoded_label = tf.image.encode_png(label_placeholder)
        else:
            assert False

        with tf.Session('') as sess:
            total_samples = len(packet['image_fnames'])
            cnt = 0
            t1 = tstart = time.time()
            for num_sample in range(total_samples):
                image = misc.imread(packet['image_fnames'][num_sample])  # uint8: [0, 255]
                label = np.atleast_3d(misc.imread(packet['label_fnames'][num_sample],
                                                  mode='P'))  # uint8: [0, 255], encoded as a grayscale png image
                # label[label == 34] = 0  # Remove troublesome license plate class
                if image.shape[:2] != packet['dims'] and not reshape:
                    print('debug:', packet['image_fnames'][num_sample])
                    print('debug:image,label:', image.dtype, image.shape, label.dtype, label.shape)
                    continue
                else:
                    cnt += 1
                if not image.shape[:2] == label.shape[:2]:
                    print(f'Unequal shapes of image {image.shape} and label {label.shape}')
                feed_dict = {image_placeholder: image,
                             label_placeholder: label,
                             raw_input_shapes: image.shape[:2] + label.shape[:2]}
                image_string, label_string = sess.run([encoded_image, encoded_label], feed_dict=feed_dict)

                example = create_tfexample(image_string, label_string, packet, num_sample)
                tfrecord_writer.write(example.SerializeToString())

                # Logging
                if num_sample % 100 == 0:
                    projected_time = (time.time() - tstart) / (num_sample + 1E-9) * (total_samples - num_sample)
                    print(f'{num_sample:6.0f}samples of {total_samples:6.0f}'
                          f'({num_sample * 100 / total_samples:5.3f}percent)'
                          f'and {time.time() - t1:5.2f} secs since last '
                          f'and {time.time() - tstart:5.2f} secs since total'
                          f'and projected time left = {projected_time:8.0f}')
                    t1 = time.time()
            print('We have bypassed %i records of %i' % (total_samples - cnt, total_samples))


def main(dataset, tvt_set, output_dims, reshape, base_path):

    assert tvt_set in ['train', 'val', 'test']

    print(f"work on dataset {dataset} and split {tvt_set} and reshape {reshape} to {output_dims}")
    if dataset == 'cityscapes':
        train_path = join(base_path, f'leftImg8bit/{tvt_set}/*/*.png')
        train_image_fnames = glob.glob(train_path)
        print('debug:train_image_fnames:', len(train_image_fnames))
        train_label_fnames = [ef.replace('leftImg8bit.png', 'gtFine_labelIds.png') for ef in train_image_fnames]
        train_label_fnames = [lf.replace('leftImg8bit/', 'gtFine/') for lf in train_label_fnames]
        print('debug:', train_image_fnames[0], train_label_fnames[0])

        image_fnames = train_image_fnames
        label_fnames = train_label_fnames

        tfrecord_fname = join(base_path, f'new_tfrecords/{tvt_set}Fine.tfrecords')
    elif dataset == 'camvid':
        train_path = join(base_path, 'LabeledApproved_full')
        train_label_fnames = glob.glob(train_path + '/*.png')
        print('debug:train_image_fnames:', len(train_label_fnames))
        train_image_fnames = [ef.replace('_L', '') for ef in train_label_fnames]
        train_image_fnames = [ef.replace('LabeledApproved_full', '701_StillsRaw_full') for ef in train_image_fnames]
        print('debug:', train_image_fnames[0], train_label_fnames[0])

        image_fnames = train_image_fnames
        label_fnames = train_label_fnames

        tfrecord_fname = join(base_path, 'tfrecords/trainFine.tfrecords')
    elif dataset == 'gta5':
        # path_out = '/home/mps/Documents/rob/datasets/gta5'
        split = loadmat('split.mat')[tvt_set+'Ids']
        split = np.squeeze(split).tolist()

        if tvt_set == 'test':
            split = split[:-4]

        image_fnames = glob.glob(join(base_path, 'images/*.png'))
        label_fnames = [ef.replace('images', 'labels') for ef in image_fnames]

        image_fnames = [image_fnames[i] for i in split]
        label_fnames = [label_fnames[i] for i in split]

        tfrecord_fname = join(base_path, f'new_tfrecords/{tvt_set}Fine.tfrecords')

    elif dataset == 'mapillary':
        print(f'start on Mapillary{tvt_set}')
        if tvt_set == 'val':
            return
        tvt_folder = 'training' if tvt_set == 'train' else 'validation'
        image_fnames = glob.glob(join(base_path, tvt_folder, 'images', '*.jpg'))
        label_fnames = [imf.replace('images', 'labels').replace('.jpg', '.png') for imf in image_fnames]

        tfrecord_fname = join(base_path, f'new_tfrecords/{tvt_set}Fine.tfrecords')

    elif dataset == 'apollo':
        if tvt_set == 'train':
            return

        image_fnames, label_fnames = get_apollo_im_label_fnames(base_path)

        tfrecord_fname = join(base_path, f'new_tfrecords/{tvt_set}Fine.tfrecords')
        maybe_makedirs(tfrecord_fname)

    elif dataset == 'wilddash':
        image_fnames = glob.glob(join(base_path, 'wd_val_01/*_100000.png'))
        label_fnames = [im_fname.replace('.png', '_labelIds.png') for im_fname in image_fnames]

        tfrecord_fname = join(base_path, 'valFine.tfrecords')
    elif dataset == 'bdd':
        image_fnames = glob.glob(join(base_path, 'images/val/*.jpg'))
        label_fnames = [im_fname.replace('images', 'labels').replace('.jpg', '_train_id.png') for im_fname in image_fnames]

        for x, y in zip(image_fnames, label_fnames):
            print(x)
            print(y)
            print('\n')
        tfrecord_fname = join(base_path, 'tfrecords_384/valFine.tfrecords')
    else:
        assert False

    with tf.python_io.TFRecordWriter(tfrecord_fname) as tfrecord_writer:
        packet = {'image_fnames': image_fnames,
                  'label_fnames': label_fnames,
                  'format': 'png',
                  'dims': output_dims}
        add_to_tfrecords(packet, tfrecord_writer, reshape=reshape)


def get_apollo_im_label_fnames(base_path):
    with open('/hdd/datasets/apolloscape/original/public_image_lists/all_val.lst') as f:
        im_label_lines = f.readlines()
        im_label_list = map(lambda im_label: tuple(im_label.replace('\n', '').split('\t')), im_label_lines)
        image_fnames, label_fnames = zip(*im_label_list)

        image_fnames = list(map(lambda rel_path: join(base_path, rel_path), image_fnames))
        label_fnames = list(map(lambda rel_path: join(base_path, rel_path), label_fnames))
    return image_fnames, label_fnames


if __name__ == '__main__':
    # base_path = '/home/mps/Documents/rob/datasets/camvid/original'
    # for tvt_set in ['train', 'val']:
    #     print('-'*80)
    #     print(f'Working on {tvt_set}')
    #     # main(dataset='cityscapes', tvt_set=tvt_set, output_dims=(384, 768),
    #     #      reshape='resize', base_path='/hdd/datasets/cityscapes')
    #     # main(dataset='gta5', tvt_set=tvt_set, output_dims=(384, 768),
    #     #      reshape='resize', base_path='/hdd/datasets/gta5')
    #     # main(dataset='mapillary', tvt_set=tvt_set, output_dims=(384, 768),
    #     #      reshape='resize', base_path='/hdd/datasets/mapillary')
    #
    #     main(dataset='apollo', tvt_set=tvt_set, output_dims=(384, 768),
    #          reshape='resize', base_path='/hdd/datasets/apolloscape/original')

    # main(dataset='wilddash', tvt_set='val', output_dims=(1080, 1920),
    #      reshape='none', base_path='/hdd/datasets/wilddash')
    main(dataset='bdd', tvt_set='val', output_dims=(384, 768),
         reshape='resize', base_path='/hdd/datasets/bdd100k/seg')


