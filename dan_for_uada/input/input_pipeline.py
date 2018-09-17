"""Input pipeline for a Semantic Segmentation model using TF Data API.

directory with images and labels
 ||
 vv
data reading, batching and preparation
(deal with shapes, dtypes, ranges and mappings)
 ||
 vv
raw input images and labels as tf.Tensor    -->  output1: (rawdata, metadata)
 ||
 vv
preprocessing
 ||
 vv
preprocessed images and labels as tf.Tensor -->  output2: (prodata, metadata)
 ||
 vv
QA: this pipeline must gurantee output rate <= 50 ms per output2
    for prodata of shape: (4 x 512 x 1024 x 3, 4 x 512 x 1024)

Directory structure: (still to be decided), for now paths provided at
otherconfig must be enough: e.g. directory paths for images and labels
and recursively scan those directories for examples (such as Cityscapes)

output1: during prediction, when plotting results the original image
  must be also available, or for saving outputs metadata such as original
  file name is needed.

output2: the actual input to Estimator.

input functions: are called by train, evaluate and predict of a
tf.estimator.Estimator instance as input_fn(config, params).
Note: only these names are checked to be passed, thus the only arguments of
input functions must be 'config' and/or 'params'.

problem definition file: a json file containing a single object with at least
the following key-value pairs:
version: version of problem definition (key reserved for later use)
lids2cids: an array of label ids to class ids mappings: each label id in the
  encoded image is mapped to a class id for classification according to this
  array. Ignoring a label is supported by class id -1. Class ids >=0.
  The validity of the mapping is upon the caller for verification. This
  pair is useful for ignoring selected annotated ids or performing category
  classification.
cids2labels: an array of class ids to labels mappings: each class id gets the
  string label of the corresponding index. Void label should be provided first.
cids2colors: an array of class ids to a 3-element array of RGB colors.
Example: parsed json to Python dictionary:
{"version":1.0,
 "comments":"label image is encoded as png with uint8 pixel values denoting
    the label id, e.g. 0:void, 1:static, 2:car, 3:human, 4:bus",
 "lids2cids":[-1,-1,1,0,2],
 "cids2labels":["void", "human", "car", "bus"],
 "cids2colors":[[0,0,0],[45,67,89],[0,0,255],[140,150,160]]}
"""

from datetime import datetime
import tensorflow as tf
from utils.utils import _replacevoids, print_tensor_info
from input.preprocess_augmentation_1 import preprocess_train, preprocess_evaluate, preprocess_predict
import glob
import numpy as np
from PIL import Image
from os.path import join

# !!! IMPORTANT: at least till TF v1.4, tensors feeding a tf.data dataset
# must have the same output shapes and types with the ones of the dataset,
# e.g. cannot have a tfrecords file with different examples shapes and feed
# it to the same dataset, to solve this the tensors can be padded with the
# proper amount of elements (see input_mapillary)


def parse_func(record):
    """
    parses the TF Records file to decoded png's
    :param record:
    :param params:
    :return:
    """
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'image/path': tf.FixedLenFeature((), tf.string, default_value=''),
        'label/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'label/format': tf.FixedLenFeature((), tf.string, default_value='png'),
        'label/path': tf.FixedLenFeature((), tf.string, default_value=''),
        'height': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64)
    }

    features = tf.parse_single_example(record, keys_to_features)

    image = tf.image.decode_png(features['image/encoded'], channels=3)
    label_dtype = tf.uint8
    label = tf.image.decode_png(features['label/encoded'], channels=1, dtype=label_dtype)
    label = tf.reshape(label, tf.convert_to_tensor([features['height'], features['width'], 1]))
    label = tf.squeeze(label)

    paths = (features['image/path'], features['label/path'])
    return image, label, paths


def prepare_data(rawimage, rawlabel, mapping, params):
    """
    Applies the preprocessing to images and labels
      - the resizing of various images and labels
      - Applies random croppings
      - Applies preprocessing functions like random colors and random blurs
    :param rawimage:
    :param rawlabel:
    :param mapping:
    :param params:
    :return:
    """
    # rawimage: TF tensor: H x W x 3, tf.uint8
    # rawlabel: TF tensor: H x W, tf.uint8/16, [0,tf.uint8/16-1]
    # images: TF tensor: Nb x hf x wf x 3, tf.float32 in [0,1)
    # labels: TF tensor: Nb x hf x wf (in case of upsampling), tf.int32, [0, Nclasses] (in case of extra void class)

    image = tf.image.convert_image_dtype(rawimage, dtype=tf.float32)
    # resize to learnable system's dimensions
    image = tf.image.resize_images(image, [params.height_network, params.width_network])

    label_for_resize = tf.to_int32(rawlabel[tf.newaxis, ..., tf.newaxis])
    label = tf.image.resize_nearest_neighbor(label_for_resize, [params.height_network, params.width_network])
    label = tf.squeeze(label, axis=[0, 3])

    label = _lids2cids(mapping, label)

    return image, label


def train_input_per_data(config, params, num_dataset):
    """
    Applies the preprocessing functions for a specific dataset
    :param config:
    :param params:
    :param num_dataset: index into the records and Nb lists
    :return:
    """
    Nb = params.Nb_list[num_dataset]
    mapping = params.training_lids2cids
    if num_dataset == 1 and hasattr(params, 'additional_lids2cids'):
        mapping = params.additional_lids2cids
    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset(params.tfrecords_list[num_dataset])
        # uncomment next line when shuffle_and_repeat becomes available
        # dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(params.Nb * 100))
        dataset = dataset.shuffle(buffer_size=params.Nb * 100)
        dataset = dataset.repeat()
        dataset = dataset.map(parse_func, num_parallel_calls=8)
        dataset = dataset.map(
            lambda image, label, paths: (paths, *prepare_data(image, label, mapping, params)))
        dataset = dataset.map(lambda paths, image, label:
                              (paths, image, label, *preprocess_train(image, label, params)), num_parallel_calls=8)
        dataset = dataset.batch(Nb)
        dataset = dataset.prefetch(Nb * 2)
        iterator = dataset.make_one_shot_iterator()
    return iterator.get_next()


def train_input(config, params):
    """
    Basis function that covers the input pipeline for training.
    It reads and decodes from TF records and applies all the pre-processing
    It returns features and labels, which are the tensors ready to use. They follow from
    an iterator on the data set API. Each fetch on them will advance the iterator by one
    :param config:
    :param params:
    :return:
    """
    """
    rawimages: Nb x hf x wf x 3, tf.float32, in [0,1]
    rawlabels: Nb x hf x wf, tf.int32, in [0,Nc-1]
    rawmetadata: Python dictionary with matadata (e.g. image shape, dtype)
    proimages: Nb x hf x wf x 3, tf.float32, in [0,1]
    prolabels: Nb x hf x wf, tf.int32, in [0,Nc-1]
    """
    # runconfig = config.runconfig
    # # otherconfig includes: train_preprocess, mappings
    # otherconfig = config.otherconfig
    # hparams = params

    # reading, mapping labels to problem from otherconfig['lids2cids'],
    # batching, preprocessing with otherconfig['train_preprocess'], output

    # no obvious use of prodata metadata for now
    with tf.variable_scope('input_pipeline'):
        values = None
        for num_dataset in range(len(params.tfrecords_list)):  # , params.camvid_tfrecords_path]:
            if values is None:
                values = train_input_per_data(config, params, num_dataset)
                values = list(values) + [num_dataset*tf.ones([params.Nb_list[num_dataset], ], dtype=tf.int32)]
            else:
                _values = list(train_input_per_data(config, params, num_dataset)) + \
                          [num_dataset*tf.ones([params.Nb_list[num_dataset], ], dtype=tf.int32)]
                values = [tf.concat((value1, value2), 0) for value1, value2 in zip(values, _values)]

        features = {'rawimages': values[1],
                    'proimages': values[3],
                    'rawimagespaths': values[0][0],
                    'rawlabelspaths': values[0][1]}
        labels = {'rawlabels': values[2],
                  'prolabels': values[4],
                  'domainlabels': values[5]}
    return features, labels


def evaluate_input(config, params):
    """
    Basis function for reading data when making evaluations
    :param config:
    :param params:
    :return:
    """
    mapping = params.evaluation_problem_def['lids2cids']
    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset(params.tfrecords_path)
        dataset = dataset.map(parse_func, num_parallel_calls=8)
        dataset = dataset.map(
            lambda image, label, paths: (paths, *prepare_data(image, label, mapping, params)))
        dataset = dataset.map(lambda paths, image, label:
                              (paths, image, label, *preprocess_evaluate(image, label, params)), num_parallel_calls=8)
        dataset = dataset.batch(params.Nb)
        dataset = dataset.prefetch(params.Nb * 10)
        iterator = dataset.make_one_shot_iterator()
        values = iterator.get_next()

    features = {'rawimages': values[1],
                'proimages': values[3],
                'rawimagespaths': values[0][0],
                'rawlabelspaths': values[0][1]}
    labels = {'rawlabels': values[2],
              'prolabels': values[4]}
    return features, labels


def extract_input(config, params, num_take=None):
    """
    Basis function for reading data when making evaluations

    Num_take indicates how many samples to use from the dataset. This could be useful in adaptation. Then you can
    specify how many samples to use for adaptation.
    :param config:
    :param params:
    :param num_take: number of samples to use from the dataset. Uses the full dataset when None
    :return:
    """
    num_prefetch = params.Nb*10
    if num_take is not None:
        num_take = max((num_take, params.Nb))
        num_prefetch = min((num_prefetch, num_take))
    mapping = params.inference_problem_def['lids2cids']
    with tf.device('/cpu:0'):
        dataset = tf.data.TFRecordDataset(params.predict_dir)
        if num_take is not None:
            dataset = dataset.take(num_take)
        dataset = dataset.repeat()
        dataset = dataset.map(parse_func, num_parallel_calls=8)
        dataset = dataset.map(
            lambda image, label, paths: (paths, *prepare_data(image, label, mapping, params)))
        dataset = dataset.map(lambda paths, image, label:
                              (paths, image, label, *preprocess_evaluate(image, label, params)), num_parallel_calls=8)
        dataset = dataset.batch(params.Nb)
        dataset = dataset.prefetch(num_prefetch)
        iterator = dataset.make_one_shot_iterator()
        values = iterator.get_next()

    features = {'rawimages': values[1],
                'proimages': values[3],
                'rawimagespaths': values[0][0],
                'rawlabelspaths': values[0][1]}
    labels = {'rawlabels': values[2],
              'prolabels': values[4]}
    return features, labels


def get_fnames_predict(path):
    """
    Gets the image filenames of all supported extensions in path recursively
    :param path:the base path where to look for media
    :return:
    """

    SUPPORTED_EXTENSIONS = ['png', 'PNG', 'jpg', 'JPG', 'jpeg', 'JPEG']

    fnames = []
    for se in SUPPORTED_EXTENSIONS:
        fnames.extend(sorted(glob.glob(join(path, '*.' + se), recursive=True)))
    return fnames


def generate_tensors_predict(params):
    """
    Yields the images as numpy arrays. Plus the paths and shapes
    :param params:
    :return:
    """
    for im_fname in get_fnames_predict(params.predict_dir):
        im = Image.open(im_fname)
        # next line is time consuming (can take up to 400ms for im of 2 MPixels)
        im_array = np.array(im)
        yield im_array, im_fname.encode('utf-8'), im_array.shape[0], im_array.shape[1]


def set_shape_predict(im, im_path, height, width):
    """
    Util function to set the shapes of the tensors dynamically
    :param im:
    :param im_path:
    :param height:
    :param width:
    :return:
    """
    im = tf.reshape(im, tf.convert_to_tensor([height, width, 3]))

    im_path.set_shape([])
    return im, im_path


def predict_input(config, params):
    with tf.device('/cpu:0'):
        dataset = tf.data.Dataset.from_generator(lambda: generate_tensors_predict(params),
                                                 output_types=(tf.uint8, tf.string, tf.int32, tf.int32))
        dataset = dataset.map(lambda im, im_path, height, width: set_shape_predict(im, im_path, height, width),
                              num_parallel_calls=8)
        dataset = dataset.map(lambda im, im_path: (im_path, im, preprocess_predict(im, params)), num_parallel_calls=8)
        dataset = dataset.batch(params.Nb)
        dataset = dataset.prefetch(params.Nb*20)
        iterator = dataset.make_one_shot_iterator()
        values = iterator.get_next()

    features = {'rawimages': values[1],
                'proimages': values[2],
                'rawimagespaths': values[0]}
    return features


# def predict_input_v2(config, params):
#     # do everything in numpy before transforming to tf.Tensor
#     # not active yet, only for benchmarking reasons
#     def gen(params):
#         for im_fname in get_fnames_predict(params.predict_dir):
#             im = Image.open(im_fname)
#             raw = np.array(im)
#             im = im.resize((params.width_network, params.height_network),
#                            resample=Image.BILINEAR)
#             # uint8 -> float32 -> [0,1] -> [-1,1]
#             im_array = np.array(im).astype(np.float32)
#             im_array /= 255
#             mean = 0.5
#             pro = im_array - mean
#             pro /= mean
#             yield raw, pro, im_fname.encode('utf-8')
#
#     output_im_shape = (params.height_network, params.width_network, 3)
#     dataset = tf.data.Dataset.from_generator(lambda: gen(params),
#                                              output_types=(tf.float32, tf.float32, tf.string),
#                                              output_shapes=(output_im_shape, output_im_shape, ()))
#     dataset = dataset.batch(params.Nb)
#     dataset = dataset.prefetch(params.Nb*20)
#     iterator = dataset.make_one_shot_iterator()
#     values = iterator.get_next()
#
#     features = {'rawimages': values[0],
#                 'proimages': values[1],
#                 'rawimagespaths': values[2]}
#     return features


def _lids2cids(lids2cids, lids):
    """
    Label ids to class ids conversion of ground truth using the lids2cids mapping.
    This function gathers cids from lids2cids according to indices from lids.
    Nl: number of labels
    Nc: number of classes

    Args:
    lids2cids: Nl, in [0, Nc-1]
    lids: H x W, tf.uint8, in [0, Nl-1]

    Returns:
    H x W, in [0, Nc-1]
    """
    # TODO: add type checking
    assert lids.dtype.is_integer, 'lids tensor must be integer.'

    return tf.gather(_replacevoids(lids2cids), lids)
