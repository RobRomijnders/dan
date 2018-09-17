import os
import numpy as np
import tensorflow as tf
import logging
from os.path import join
from glob import glob

log = logging.getLogger('semantic_segmentation')


def get_all_ckpt_paths(direc):
    all_ckpt_paths = map(lambda x: x.rstrip('.index'), glob(
        join(direc, 'model.ckpt-*.index')))
    # Then sort them and select the final checkpoints

    def sort_checkpoints_key(checkpoint_name):
        checkpoint_num = checkpoint_name.split('-')[-1]
        assert checkpoint_num.isdigit()
        return int(checkpoint_num)
    all_ckpt_paths = list(sorted(all_ckpt_paths, key=sort_checkpoints_key))
    assert len(all_ckpt_paths) > 0, f"Found no checkpoints in the directory {direc}"
    return all_ckpt_paths


def print_tensor_info(tensor):
    print(f"debug:{tensor.op.name}: {tensor.get_shape().as_list()} {tensor.dtype}")


def get_unique_variable_by_name_without_creating(name):
    variables = [v for v in tf.global_variables() + tf.local_variables() if name == v.op.name]
    assert len(variables) == 1, f"Found {len(variables)} variables for name {name}."
    return variables[0]


def get_unique_tensor_by_name_without_creating(name):
    tensors = [t for t in tf.contrib.graph_editor.get_tensors(tf.get_default_graph()) if name == t.name]
    assert len(tensors) == 1, f"Found {len(tensors)} tensors."
    return tensors[0]


def get_saveable_objects_list(graph):
    return graph.get_collection(tf.GraphKeys.GLOBAL_VARIABLES) + graph.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS)


def ids2image(ids, palette):
    # ids: Nb x H x W, with elements in [0,K-1]
    # palette: K x 3, tf.uint8
    # returns: Nb x H x W x 3, tf.uint8

    # TODO: add type checking
    return tf.gather_nd(palette, tf.expand_dims(ids, axis=-1))


# def convert_ids(label, mode, params):
#   # label: TF tensor: H x W, tf.int32, [0,tf.int32-1]
#   # mode: one of tf.estimator.ModeKeys
#   # mappings: python list mapping ids in label to classes (using -1 for void class)

#   if mode is tf.estimator.ModeKeys.TRAIN:
#     mappings = params.training_problem_def['lids2cids']
#   elif mode is tf.estimator.ModeKeys.EVAL:
#     mappings = params.evaluation_problem_def['lids2cids']
#   elif mode is tf.estimator.ModeKeys.PREDICT:
#     mappings = params.inference_problem_def['lids2cids']
#   else:
#     assert False, f'mode {mode} not supported.'

#   return ids2image(label, _replacevoids(mappings))

def almost_equal(num1, num2, error=10 ** -3):
    return abs(num1 - num2) <= error


def _replacevoids(mappings):
    # previous code replaced voids with max id + 1
    max_m = max(mappings)
    return [m if m != -1 else max_m + 1 for m in mappings]
    # replace void ids (-1) if exist with Nclasses + 1
    # new code not debugged
    # Nclasses = max(config['mappings']) + 1 + (1 if -1 in config['mappings'] else 0)
    # return [m if m!=-1 else Nclasses for m in mappings]


# safe_div from tensorflow/python/ops/losses/losses_impl.py
def safe_div(num, den, name="safe_div"):
    """Computes a safe divide which returns 0 if the den is zero.
    Note that the function contains an additional conditional check that is
    necessary for avoiding situations where the loss is zero causing NaNs to
    creep into the gradient computation.
    Args:
      num: An arbitrary `Tensor`.
      den: `Tensor` whose shape matches `num` and whose values are
        assumed to be non-negative.
      name: An optional name for the returned op.
    Returns:
      The element-wise value of the num divided by the den.
    """
    return tf.where(tf.greater(den, 0),
                    tf.div(num,
                           tf.where(tf.equal(den, 0),
                                    tf.ones_like(den), den)),
                    tf.zeros_like(num),
                    name=name)


def print_metrics_from_confusion_matrix(cm, labels=None, printfile=None, summary=False):
    """

    :param cm: confusion matrix
    :param labels: python list of label names
    :param printfile: (opened) file to print the metrics to
    :param summary: if printfile is not None, prints only a summary of metrics to file
    :return:
    """

    # sanity checks
    assert isinstance(cm, np.ndarray), 'Confusion matrix must be numpy array.'
    cms = cm.shape
    assert all([cm.dtype in [np.int32, np.int64],
                cm.ndim == 2,
                cms[0] == cms[1],
                not np.any(np.isnan(cm))]), (
        f"Check print_metrics_from_confusion_matrix input requirements. "
        f"Input has {cm.ndim} dims, is {cm.dtype}, has shape {cms[0]}x{cms[1]} "
        f"and may contain NaNs.")
    if not labels:
        labels = ['unknown'] * cms[0]
    assert len(labels) == cms[0], (
        f"labels ({len(labels)}) must be enough for indexing confusion matrix ({cms[0]}x{cms[1]}).")
    # assert os.path.isfile(printfile), 'printfile is not a file.'

    # metric computations
    global_accuracy = np.trace(cm) / np.sum(cm) * 100
    # np.sum(cm,1) can be 0 so some accuracies can be nan
    accuracies = np.diagonal(cm) / (np.sum(cm, 1) + 1E-9) * 100
    # denominator can be zero only if #TP=0 which gives nan, trick to avoid it
    inter = np.diagonal(cm)
    union = np.sum(cm, 0) + np.sum(cm, 1) - np.diagonal(cm)
    ious = inter / np.where(union > 0, union, np.ones_like(union)) * 100
    notnan_mask = np.logical_not(np.isnan(accuracies))
    mean_accuracy = np.mean(accuracies[notnan_mask])
    mean_iou = np.mean(ious[notnan_mask])

    # reporting
    log_string = "\n"
    log_string += f"Global accuracy: {global_accuracy:5.2f}\n"
    log_string += "Per class accuracies (nans due to 0 #Trues) and ious (nans due to 0 #TPs):\n"
    for k, v in {l: (a, i, m) for l, a, i, m in zip(labels, accuracies, ious, notnan_mask)}.items():
        log_string += f"{k:<30s}  {v[0]:>5.2f}  {v[1]:>5.2f}  {'' if v[2] else '(ignored in averages)'}\n"
    log_string += f"Mean accuracy (ignoring nans): {mean_accuracy:5.2f}\n"
    log_string += f"Mean iou (ignoring accuracies' nans but including ious' 0s): {mean_iou:5.2f}\n"
    log_string += f"\n Latex formatted \n"
    log_string += '%s & %.1f' % (' & '.join([str('%.1f' % num) for num in ious]), mean_iou)

    if True:
        log.info(log_string)

    if printfile:
        if summary:
            printfile.write(log_string)
        else:
            print(f"{global_accuracy:>5.2f}",
                  f"{mean_accuracy:>5.2f}",
                  f"{mean_iou:>5.2f}",
                  accuracies,
                  ious,
                  file=printfile)


def print_all_metrics(all_metrics, labels=None, printfile=None):
    """

    :param all_metrics: list with all metrics, resulting from system.evaluate()
    :param labels: python list of label names
    :param printfile: (opened) file to print the metrics to
    :return:
    """
    for metrics in all_metrics:
        cm = metrics['confusion_matrix']
        step = metrics['global_step']

        # sanity checks
        cm_shape = cm.shape
        assert all([cm.dtype in [np.int32, np.int64],
                    cm.ndim == 2,
                    cm_shape[0] == cm_shape[1],
                    not np.any(np.isnan(cm))]), (
            f"Check print_metrics_from_confusion_matrix input requirements. "
            f"Input has {cm.ndim} dims, is {cm.dtype}, has shape {cm_shape[0]}x{cm_shape[1]} "
            f"and may contain NaNs.")
        if not labels:
            labels = ['unknown'] * cm_shape[0]
        assert len(labels) == cm_shape[0], (
            f"labels ({len(labels)}) must be enough for indexing confusion matrix ({cm_shape[0]}x{cm_shape[1]}).")

        # metric computations
        global_accuracy = np.trace(cm) / np.sum(cm) * 100
        # np.sum(cm,1) can be 0 so some accuracies can be nan
        accuracies = np.diagonal(cm) / (np.sum(cm, 1) + 1E-9) * 100
        # denominator can be zero only if #TP=0 which gives nan, trick to avoid it
        inter = np.diagonal(cm)
        union = np.sum(cm, 0) + np.sum(cm, 1) - np.diagonal(cm)
        ious = inter / np.where(union > 0, union, np.ones_like(union)) * 100
        notnan_mask = np.logical_not(np.isnan(accuracies))
        mean_accuracy = np.mean(accuracies[notnan_mask])
        mean_iou = np.mean(ious[notnan_mask])

        # reporting
        log_string = "\n" * 3
        log_string += f"Global step: {step} \n"
        log_string += f"Global accuracy: {global_accuracy:5.2f}\n"
        log_string += "Per class accuracies (nans due to 0 #Trues) and ious (nans due to 0 #TPs):\n"
        for k, v in {l: (a, i, m) for l, a, i, m in zip(labels, accuracies, ious, notnan_mask)}.items():
            log_string += f"{k:<30s}  {v[0]:>5.2f}  {v[1]:>5.2f}  {'' if v[2] else '(ignored in averages)'}\n"
        log_string += f"Mean accuracy (ignoring nans): {mean_accuracy:5.2f}\n"
        log_string += f"Mean iou (ignoring accuracies' nans but including ious' 0s): {mean_iou:5.2f}\n"
        log_string += f"\n Latex formatted \n"
        log_string += '%s & %.1f' % (' & '.join([str('%.1f' % num) for num in ious]), mean_iou)
        log_string += "\n"

        printfile.write(log_string)


def split_path(path):
    # filepath = <head>/<tail>
    # filepath = <head>/<root>.<ext[1:]>
    head, tail = os.path.split(path)
    root, ext = os.path.splitext(tail)
    return head, root, ext[1:]


def _validate_problem_config(config):
    # config is a json file that contains problem configuration
    #   it should contain at least the following fields:
    #     version: version of config json
    #     mappings: index:label id -> class id
    #     labels: index:class id -> label string
    #     palettes: index:class id -> RGB color
    #     idspalettes: index:class id -> label id (temporary field:
    #       can be infered from mappings)

    # check if all required keys are present
    mandatory_keys = ['version', 'mappings', 'labels', 'palettes', 'idspalettes']
    assert all([k in config.keys() for k in mandatory_keys]), (
        f"problem config json must also have {set(mandatory_keys)-set(config.keys())}")
    # check version
    assert config['version'] == 2.0

    # -- type and value checking
    # check mappings validity
    assert all([isinstance(m, int) and m >= -1 for m in config['mappings']]), (
        'Only integers and >=-1 are allowed in mappings.')
    # check labels validity
    assert all([isinstance(l, str) for l in config['labels']]), (
        'Only string labels are allowed in labels.')
    # check palettes validity
    assert all([isinstance(c, int) and 0 <= c <= 255
                for p in config['palettes'] for c in p])

    # -- length checking
    # Nclasses = count_non_i(config['mappings'], -1) + (1 if -1 in config['mappings'] else 0)
    # +1 since class ids start from 0 by convention, if -1 (void) exist another
    #   label, palette and idspalette are needed
    # TODO: check also problemXXTOproblemXX fields if they exist
    Nclasses = max(config['mappings']) + 1 + (1 if -1 in config['mappings'] else 0)
    lens = list(map(len, [config['labels'], config['palettes']]))
    assert all([length == Nclasses for length in lens]), (
        f"Lengths of labels and palettes ({lens}) are not compatible "
        f"with number of classes ({Nclasses}).")
    assert all([len(p) == 3 for p in config['palettes']])


def count_non_i(int_lst, i):
    # counts the number of integers not equal to i in the integer list int_lst

    # assertions
    assert isinstance(int_lst, list), 'int_lst is not a list.'
    assert all([isinstance(e, int) for e in int_lst]), 'Not integer int_list.'
    assert isinstance(i, int), 'Not integer i.'

    # implementation
    return len(list(filter(lambda k: k != i, int_lst)))


def maybe_makedirs(filename, force_dir=False):
    """
    Maybe makes the necessary directories for filename

    By default, it starts at the dirname of filepath. If force_dir, then it forces that filename is itself a dirname
    :param filename:
    :param force_dir:
    :return:
    """
    if force_dir:
        filename = filename.rstrip('/') + '/'
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
