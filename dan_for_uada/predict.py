"""Example of how to use Semantic Segmentation system for prediction.
"""

import sys
import os
import collections
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from system_factory import SemanticSegmentation
from input.input_pipeline import extract_input, predict_input
from model.model import model as model_fn
from utils.utils_argparse import SemanticSegmentationArguments, set_bn_cancellation
from utils.utils import split_path, maybe_makedirs
from utils.util_run import setup_logger
from os.path import join, basename
from utils.util_plot import palette_cid_int8


def main(argv):
    ssargs = SemanticSegmentationArguments()
    ssargs.add_predict_arguments()
    args = ssargs.parse_args(argv)

    _add_extra_args(args)

    # vars(args).items() returns (key,value) tuples from args.__dict__
    # and sorted uses first element of tuples to sort
    args_dict = collections.OrderedDict(sorted(vars(args).items()))
    print(args_dict)

    settings = args

    logger = setup_logger(args.log_dir, 'predict')
    logger.warning('Hello prediction')

    if os.path.isdir(args.predict_dir):
        predict_fn = predict_input
    else:
        predict_fn = extract_input

    system = SemanticSegmentation({'predict': predict_fn}, model_fn, settings)

    start_for_total = datetime.now()
    for count, outputs in enumerate(system.get_predictions()):
        logger.debug(count)
        decs = outputs['decisions']
        rawimagepath = str(outputs['rawimagespaths'])

        filename_out = join(args.results_dir, split_path(rawimagepath)[1] + '.png')

        if False:
            decs = np.take(args.training_problem_def['cids2lids'], decs).astype(np.int32)

        if False:
            labels_color = np.take(palette_cid_int8, decs, axis=0).astype(np.uint8)

        Image.fromarray(decs).save(filename_out)

    print('\nTotal time:', datetime.now() - start_for_total)


def _add_extra_args(args):
    # disable regularizer and set batch_norm_decay to random value for with... to work
    args.batch_norm_istraining = False
    args.regularization_weight = 0.0
    args.batch_norm_decay = 1.0

    # prediction keys for SemanticSegmentation.predict(), predictions are defined in model.py
    args.predict_keys = ['decisions', 'rawimages', 'rawimagespaths']

    set_bn_cancellation(args)
    args.Nb_list = [args.Nb]

    # Get experiment name
    args.experiment_name = basename(args.log_dir.rstrip('/'))

    # args.output_dir = join(args.log_dir, 'predictions_apollo/')
    maybe_makedirs(args.results_dir, force_dir=True)


if __name__ == '__main__':
    main(sys.argv[1:])
