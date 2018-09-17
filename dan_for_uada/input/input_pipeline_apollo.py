import tensorflow as tf
import sys, glob
from os.path import join, split, realpath
import functools

sys.path.append(split(split(realpath(__file__))[0])[0])
import numpy as np
from input.preprocess_augmentation_1 import preprocess_evaluate
from utils.utils import _replacevoids


def _load_and_decode(data_location, im_la_files):
    """
    Loads and decodes images and labels from a location specified as a string

    Args:
        data_location: a String or tf.string with the location of the Apolloscape dataset
        im_la_files: a tf.string with the location of the image and the label, separated by a tab.

    Returns:

    """
    data_location = tf.cast(data_location, tf.string)

    im_la_files = tf.cast(im_la_files, tf.string)
    im_la_files_split = tf.string_split([im_la_files], '\t')
    im_file = im_la_files_split.values[0]
    la_file = im_la_files_split.values[1]

    im_string = tf.read_file(tf.string_join([data_location, im_file]))
    im_dec = tf.image.decode_jpeg(im_string)

    la_string = tf.read_file(tf.string_join([data_location, la_file]))
    la_dec = tf.image.decode_png(la_string)[..., 0]

    return im_dec, la_dec, im_file, la_file


def _evaluate_preprocess(image, label, params):
    _SIZE_FEATURE_EXTRACTOR = (params.height_network, params.width_network)

    ## prepare
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    label = tf.gather(tf.cast(_replacevoids(params.evaluation_problem_def['lids2cids']), tf.int32), tf.to_int32(label))

    ## preprocess
    proimage = tf.image.resize_images(image, _SIZE_FEATURE_EXTRACTOR)
    prolabel = tf.image.resize_images(label[..., tf.newaxis],
                                      _SIZE_FEATURE_EXTRACTOR,
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]

    proimage, _ = preprocess_evaluate(proimage)

    print('debug: proimage, prolabel', proimage, prolabel)

    return image, label, proimage, prolabel


def _evaluate_parse_and_preprocess(im_la_files, data_location, params):
    image, label, im_path, la_path = _load_and_decode(data_location, im_la_files)
    image, label, proimage, prolabel = _evaluate_preprocess(image, label, params)

    return image, label, proimage, prolabel, im_path, la_path


def evaluate_input(config, params):
    del config

    data_location = params.dataset_directory
    filenames_list = params.filelist_filepath
    filenames_string = tf.cast(filenames_list, tf.string)

    dataset = tf.data.TextLineDataset(filenames=filenames_string)

    dataset = dataset.map(
        functools.partial(_evaluate_parse_and_preprocess, data_location=data_location, params=params),
        num_parallel_calls=30)
    # IMPORTANT: if Nb > 1, then shape of dataset elements must be the same
    dataset = dataset.batch(params.Nb)

    def _grouping(rim, rla, pim, pla, imp, lap):
        # group dataset elements as required by estimator
        features = {
            'rawimages': rim,
            'proimages': pim,
            'rawimagespaths': imp,
            'rawlabelspaths': lap,
        }
        labels = {
            'rawlabels': rla,
            'prolabels': pla,
        }

        return features, labels

    dataset = dataset.map(_grouping, num_parallel_calls=30)
    dataset = dataset.prefetch(10)

    return dataset
