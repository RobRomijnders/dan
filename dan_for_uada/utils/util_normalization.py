from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.layers.python.layers import layers
from tensorflow import concat
import tensorflow as tf
from tensorflow.contrib.slim import l2_regularizer


import logging

log = logging.getLogger('semantic_segmentation')


@add_arg_scope
def split_batch_norm(inputs, Nb_list, *args, **kwargs):
    log.debug('You are splitting the batchnorm layers')
    if len(Nb_list) > 1:
        with tf.variable_scope('bn_split'):
            source_output = layers.batch_norm(inputs=inputs[:Nb_list[0]], *args, **kwargs)
        with tf.variable_scope('bn_split', reuse=True):
            target_output = layers.batch_norm(inputs=inputs[Nb_list[0]:], *args, **kwargs)
        # with tf.variable_scope("split_bn_target"):
        #     # TODO initialize the layers of target with also parameters from the ResNET checkpoints
        #     kwargs.update({'param_regularizers': {'beta': l2_regularizer(0.00017)}})
        #     target_output = layers.batch_norm(inputs=inputs[Nb_list[0]:], *args, **kwargs)
        return concat((source_output, target_output), axis=0)
    else:
        with tf.variable_scope("bn_split"):
            return layers.batch_norm(inputs=inputs, *args, **kwargs)
