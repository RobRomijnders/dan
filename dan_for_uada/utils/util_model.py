from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.layers.python.layers import regularizers
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops
from tensorflow.contrib import layers as layers_lib
# from utils.util_normalization import custom_normalization
from utils.utils_batch_norm import batch_norm as custom_batch_norm
from utils.util_normalization import split_batch_norm
import logging
from tensorflow.contrib.slim import l2_regularizer


log = logging.getLogger('semantic_segmentation')


def resnet_arg_scope_custom(weight_decay=0.0001,
                            batch_norm_decay=0.997,
                            batch_norm_epsilon=1e-9,
                            do_scale=True,
                            normalization_mode='batch',
                            is_training=True,
                            Nb_list=None,
                            regularize_extra=1):
    """Defines the default ResNet arg scope.

    TODO(gpapan): The batch-normalization related default values above are
      appropriate for use in conjunction with the reference ResNet models
      released at https://github.com/KaimingHe/deep-residual-networks. When
      training ResNets from scratch, they might need to be tuned.

    Args:
      weight_decay: The weight decay to use for regularizing the model.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      batch_norm_epsilon: Small constant to prevent division by zero when
        normalizing activations by their variance in batch normalization.
      do_scale: If True, uses an explicit `gamma` multiplier to scale the
        activations in the batch normalization layer.
      normalization_mode: in which mode to normalize (batch, layer, instance or none)
      is_training: indicate to train the batch norm parameters and update the moving averages or not

    Returns:
      An `arg_scope` to use for the resnet models.
    """
    assert normalization_mode in ['batch', 'none', 'None', None, 'custombatch', 'custombatch3'], \
        "no normalization mode %s" % normalization_mode
    normalization_params = {'scale': do_scale,
                            'updates_collections': ops.GraphKeys.UPDATE_OPS,
                            'decay': batch_norm_decay,
                            'is_training': is_training}

    if normalization_mode == 'batch':
        normalization_params.update({'epsilon': batch_norm_epsilon})
        normalizer_function = layers.batch_norm
    elif normalization_mode is None or normalization_mode in ['none', 'None']:
        log.debug('Normalization: you are doing no normalization')
        normalizer_function = None
    elif normalization_mode == 'custombatch':
        normalization_params.update({'epsilon': batch_norm_epsilon,
                                     'Nb_list': Nb_list})
        if regularize_extra > 1:
            normalization_params.update({'param_regularizers': {'beta': l2_regularizer(0.00017)}})
            log.debug('You are regularizing the beta coefficients of the batch norm')
        normalizer_function = custom_batch_norm
    elif normalization_mode == 'custombatch3':
        normalization_params.update({'epsilon': batch_norm_epsilon,
                                     'Nb_list': Nb_list})
        normalizer_function = split_batch_norm
    else:
        assert False

    if normalizer_function is not None:
        with arg_scope(
                    [layers_lib.conv2d],
                    weights_regularizer=regularizers.l2_regularizer(weight_decay),
                    weights_initializer=initializers.variance_scaling_initializer(),
                    activation_fn=nn_ops.relu,
                    normalizer_fn=normalizer_function,
                    normalizer_params=normalization_params):
            with arg_scope([normalizer_function], **normalization_params):
                # The following implies padding='SAME' for pool1, which makes feature
                # alignment easier for dense prediction tasks. This is also used in
                # https://github.com/facebook/fb.resnet.torch. However the accompanying
                # code of 'Deep Residual Learning for Image Recognition' uses
                # padding='VALID' for pool1. You can switch to that choice by setting
                # tf.contrib.framework.arg_scope([tf.contrib.layers.max_pool2d], padding='VALID').
                with arg_scope([layers.max_pool2d], padding='SAME') as arg_sc:
                    return arg_sc
    else:
        with arg_scope(
                    [layers_lib.conv2d],
                    weights_regularizer=regularizers.l2_regularizer(weight_decay),
                    weights_initializer=initializers.variance_scaling_initializer(),
                    activation_fn=nn_ops.relu,
                    normalizer_fn=normalizer_function,
                    normalizer_params=normalization_params):
            # The following implies padding='SAME' for pool1, which makes feature
            # alignment easier for dense prediction tasks. This is also used in
            # https://github.com/facebook/fb.resnet.torch. However the accompanying
            # code of 'Deep Residual Learning for Image Recognition' uses
            # padding='VALID' for pool1. You can switch to that choice by setting
            # tf.contrib.framework.arg_scope([tf.contrib.layers.max_pool2d], padding='VALID').
            with arg_scope([layers.max_pool2d], padding='SAME') as arg_sc:
                return arg_sc
