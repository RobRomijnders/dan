"""Example of how to use Semantic Segmentation system for evaluation.
"""

import collections
import sys
import os
import numpy as np
import time

from system_factory import SemanticSegmentation
from input.input_pipeline import evaluate_input as eval_fn_original
from input.input_pipeline_apollo import evaluate_input as eval_fn_apollo
from model.model import model as model_fn
from utils.utils_argparse import SemanticSegmentationArguments, set_bn_cancellation
from utils.utils import maybe_makedirs, print_all_metrics
from utils.util_run import setup_logger
np.set_printoptions(formatter={'float': '{:>5.2f}'.format}, nanstr=u'nan', linewidth=10000)


def main(argv):
    t_start = time.time()
    # Parse the arguments
    ssargs = SemanticSegmentationArguments()
    ssargs.add_evaluate_arguments()
    args = ssargs.parse_args(argv)

    _add_extra_args(args)

    # vars(args).items() returns (key,value) tuples from args.__dict__
    # and sorted uses first element of tuples to sort
    args_dict = collections.OrderedDict(sorted(vars(args).items()))
    params = args

    # Set up the logger
    logger = setup_logger(args.log_dir, 'eval')
    logger.warning('Hello evaluation')
    logger.debug('\n'.join(('%30s : %s' % (key, value) for key, value in args_dict.items())))

    # Prepare the labels to be used for printing
    labels = params.evaluation_problem_def['cids2labels']
    void_exists = -1 in params.evaluation_problem_def['lids2cids']
    labels = labels[:-1] if void_exists else labels

    eval_fn = eval_fn_original

    system = SemanticSegmentation({'eval': eval_fn}, model_fn, params)
    all_metrics = system.evaluate()

    # Print and save the confusion matrix
    output_filename = params.confusion_matrix_filename if params.confusion_matrix_filename \
        else os.path.join(system.eval_res_dir, 'confusion_matrix.txt')
    maybe_makedirs(output_filename)
    with open(output_filename, 'w') as f:
        print_all_metrics(all_metrics, labels, printfile=f)
        print('print confusion matrix in %s' % output_filename)
    print(f'Took {time.time() - t_start} seconds to run')


def _add_extra_args(args):
    """
    adds additional arguments that need to be calculated after the parsing
    :param args: the argparse object
    :return:
    """
    # number of examples per epoch
    args.num_examples = int(args.Neval *
                            args.height_network // args.height_feature_extractor *
                            args.width_network // args.width_feature_extractor)
    args.num_batches_per_epoch = int(args.num_examples / args.Nb)
    args.num_eval_steps = int(args.num_batches_per_epoch * 1)  # 1 epoch

    # disable regularizer and set batch_norm_decay to random value
    # temp solution so as with blocks to work
    args.batch_norm_istraining = False
    args.regularization_weight = 0.0
    args.batch_norm_decay = 1.0

    # Set a list of batchsizes for multiple domain training
    args.Nb_list = [args.Nb]

    # Infer batch norm settings from the settings.txt
    args = set_bn_cancellation(args)
    # args.custom_normalization_mode = 'custombatch'

    # force disable XLA, since there is an internal TF error till at least r1.4
    # TODO: remove this when error is fixed
    args.enable_xla = False


if __name__ == '__main__':
    main(sys.argv[1:])
