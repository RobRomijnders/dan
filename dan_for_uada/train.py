"""Example of how to use Semantic Segmentation system for training.
"""

import sys
import collections
import os

import tensorflow as tf

from system_factory import SemanticSegmentation
from input.input_pipeline import train_input as train_fn
from model.model import model as model_fn
from utils.utils_argparse import SemanticSegmentationArguments
from utils.util_run import zipit, setup_logger, maybe_remove_log_dir
from os.path import join
import numpy as np


def main(argv):
    # Parse all arguments
    ssargs = SemanticSegmentationArguments()
    ssargs.add_train_arguments()
    args = ssargs.parse_args(argv)
    maybe_remove_log_dir(args.log_dir)

    # Set up the logger
    logger = setup_logger(args.log_dir, 'train')
    logger.warning('Hello training')

    _add_extra_args(args, logger)

    # vars(args).items() returns (key,value) tuples from args.__dict__
    # and sorted uses first element of tuples to sort
    args_dict = collections.OrderedDict(sorted(vars(args).items()))
    logger.debug('\n'.join(('%30s : %s' % (key, value) for key, value in args_dict.items())))

    params = args

    system = SemanticSegmentation({'train': train_fn}, model_fn, params)

    if not tf.gfile.Exists(args.log_dir):
        tf.gfile.MakeDirs(args.log_dir)

    # Zip all code to the log_dir. This is for using multiple experiment setups
    code_direc = os.path.dirname(__file__)
    logger.info('Zip all code from %s' % code_direc)
    zipit(code_direc, join(args.log_dir, 'all_code.zip'))

    try:
        system.train()
    except Exception as e:
        logger.error(e)
        raise Exception(e)


def _add_extra_args(args, logger):
    """
    Lots of calculation on the number of steps to make.

    As arguments, the user defines the batch size and number of epochs to train.
    Three things to keep in mind

      * For multiple domain training, an epoch is defined by the smallest of training sizes
      * For multiple domain training, the batch_size argument defines the batch_size per dataset
      * When switching the train op, this effectively halves the number of epochs, and we correct for that by
      doubling the Ne argument

    :param args:
    :return:
    """
    # extra args for train
    # in case of patch-training of feature extractor *_network != *_feature_extractor and the
    # effective number of examples increase
    if args.switch_train_op and args.tfrecords_path_add:
        logger.debug('We double the number of epochs because of the alternating training')
        args.Ne *= 2

    args.num_examples = int(args.Ntrain *
                            args.height_network // args.height_feature_extractor *
                            args.width_network // args.width_feature_extractor)  # per epoch
    args.num_batches_per_epoch = int(args.num_examples / args.Nb)
    args.num_training_steps = int(args.num_batches_per_epoch * args.Ne)  # per training

    # Make the learning rate schedule
    if args.learning_rate_schedule == 'schedule1':
        lr_boundaries_epochs = 1 / 17 * np.array([8, 7, 2])  # best: 8, 7, 2, 2, 1
    elif args.learning_rate_schedule == 'schedule2':
        assert False
    else:
        assert False, 'No such learning rate schedule.'
    num_decay_steps = len(lr_boundaries_epochs)

    lr_boundaries = args.Ne * np.cumsum(lr_boundaries_epochs * args.num_batches_per_epoch)
    args.lr_boundaries = np.round(lr_boundaries).astype(np.int32).tolist()

    args.lr_values = [args.learning_rate_initial * args.learning_rate_decay ** i for i in
                      range(num_decay_steps + 1)]

    logger.debug(f'lr values {args.lr_values}, lr_boundaries {args.lr_boundaries} '
                 f'for total steps {args.num_training_steps}')

    # Make the record paths and the batchsizes iterable to use for multiple dataset training
    if args.tfrecords_path_add:
        args.tfrecords_list = [args.tfrecords_path, args.tfrecords_path_add]
        args.Nb_list = [args.Nb] * 2
    else:
        args.tfrecords_list = [args.tfrecords_path]
        args.Nb_list = [args.Nb]


if __name__ == '__main__':
    main(sys.argv[1:])
