"""User-defined dense semantic segmentation model.
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from model.feature_extractor import feature_extractor
from utils.utils import almost_equal
from utils.utils_batch_norm import batch_norm as custom_batch_norm
import logging
log = logging.getLogger('semantic_segmentation')


def model(mode, features, labels, config, params):
    """
    Build the full model, from the input images to the output predictions
    :param mode:
    :param features:
    :param labels:
    :param config:
    :param params:
    :return:
    """
    # build the feature extractor
    features, end_points = feature_extractor(mode, features, labels, config, params)

    if mode == tf.estimator.ModeKeys.TRAIN or hasattr(params, 'get_domain_logits'):
            # TODO(rob) fix this temporary hack
            # Only use domain classifier in training mode
        if params.switch_train_op:  # Only do domain classifier if we end up using it :p
            with tf.variable_scope('domain_classifier'):
                # General calculation for how many representations we have in the output of representation learner
                num_feature_vecs = int(
                    params.height_feature_extractor *
                    params.width_feature_extractor / params.stride_feature_extractor ** 2)
                features_stack = tf.reshape(features, shape=(sum(params.Nb_list) *
                                                             num_feature_vecs, params.feature_dims_decreased))

                # Squeeze the domain logits because it is just simpler
                domain_logits = tf.squeeze(domain_classifier(features_stack, params), axis=1)

    regularizer = slim.l2_regularizer(params.regularization_weight)
    with tf.variable_scope('segmentation_layer'):
        # Add one segmentation layer after the representations. We found this improves use of adversarial adaptation
        h1 = slim.conv2d(features,
                         kernel_size=3,
                         normalizer_fn=custom_batch_norm,
                         normalizer_params={'decay': params.batch_norm_decay,
                                            'is_training': params.batch_norm_istraining,
                                            'Nb_list': params.Nb_list},
                         num_outputs=40,
                         activation_fn=tf.nn.selu,
                         scope='segmentation_layer',
                         weights_regularizer=regularizer)

    # - create logits, probabilities and top-1 decisions
    # -   First the logits are created and then upsampled for memory efficiency.
    with tf.variable_scope(
            'softmax_classifier',
            initializer=slim.initializers.variance_scaling_initializer(),
            regularizer=slim.regularizers.l2_regularizer(params.regularization_weight)):
        logits = slim.conv2d(h1,
                             num_outputs=params.training_Nclasses,
                             kernel_size=1,
                             activation_fn=None,
                             scope='logits',
                             weights_regularizer=regularizer)
        log.debug(f'logits: {logits.op.name}, {logits.shape}')
        logits = _create_upsampler(logits, params)
        log.debug(f'upsampled logits: {logits.op.name},{logits.shape}')
        # during training probs and decs are used only for summaries
        # (not by train op) and thus not need to be computed in every step (GPU)
        # WARNING: next branch needed because if tf.device(None) is used,
        #   outer device allocation is not possible
        if mode == tf.estimator.ModeKeys.TRAIN:
            with tf.device('/cpu:0'):
                probs = tf.nn.softmax(logits, name='probabilities')
                decs = tf.cast(tf.argmax(probs, 3), tf.int32, name='decisions')
        else:
            probs = tf.nn.softmax(logits, name='probabilities')
            decs = tf.cast(tf.argmax(probs, 3), tf.int32, name='decisions')
        log.debug(f'decisions:{decs.op.name},{decs.shape},{decs}')

    # -- model outputs groupped as predictions of the Estimator
    # WARNING: 'decisions' key is used internally so it must exist for now..
    predictions = {'logits': logits,
                   'probabilities': probs,
                   'decisions': decs}
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions['representations'] = features
        if hasattr(params, 'get_domain_logits'):
            # TODO(rob) fix this temporary hack
            predictions['domain_logits'] = tf.reshape(domain_logits, shape=[4, 48, 96])
    if mode == tf.estimator.ModeKeys.TRAIN and params.switch_train_op:
        predictions['domain_logits'] = domain_logits

    return features, end_points, predictions


def _create_upsampler(bottom, params):
    # upsample bottom depthwise to reach feature extractor output dimensions
    # bottom: Nb x hf//sfe x hf//sfe x C
    # upsampled: Nb x hf x hf x C
    # TODO: Does upsampler needs regularization??
    # TODO: align_corners=params.enable_xla: XLA implements only align_corners=True for now,
    # change it when XLA implements all

    C = bottom.shape[-1]
    spat_dims = np.array(bottom.shape.as_list()[1:3])
    hf, wf = params.height_feature_extractor, params.width_feature_extractor
    # WARNING: Resized images will be distorted if their original aspect ratio is not the same as size
    # (only in the case of bilinear resizing)
    if params.upsampling_method != 'no':
        assert almost_equal(spat_dims[0] / spat_dims[1],
                            params.height_feature_extractor / params.width_feature_extractor,
                            10 ** -1), (
            f"Resized images will be distorted if their original aspect ratio is "
            f"not the same as size: {spat_dims[0],spat_dims[1]}, {hf,wf}.")
    with tf.variable_scope('upsampling'):
        if params.upsampling_method == 'no':
            upsampled = bottom
        elif params.upsampling_method == 'bilinear':
            # TODO(rob) do we still use this option, ever?
            upsampled = tf.image.resize_images(bottom, [hf, wf], align_corners=params.enable_xla)
        elif params.upsampling_method == 'hybrid':
            # TODO(rob) it seems we always use this one??
            # composite1: deconv upsample twice and then resize
            assert params.stride_feature_extractor in (4, 8, 16), 'stride_feature_extractor must be 4, 8 or 16.'
            upsampled = slim.conv2d_transpose(inputs=bottom,
                                              num_outputs=C,
                                              kernel_size=2 * 2,
                                              stride=2,
                                              padding='SAME',
                                              activation_fn=None,  # No activation function necessary after upsampling
                                              weights_initializer=slim.variance_scaling_initializer(),
                                              weights_regularizer=slim.l2_regularizer(params.regularization_weight))
            upsampled = tf.image.resize_images(upsampled, [hf, wf], align_corners=params.enable_xla)
        else:
            upsampled = None
            raise ValueError('No such upsampling method.')

    return upsampled


def domain_classifier(features, params):
    features = add_diff_noise(features)

    if params.dom_class_type == 1:
        regularizer = slim.l2_regularizer(params.regularization_weight)
        log.debug('Use alternative domain classifier')
        h1 = tf.layers.dense(features,
                             200,
                             activation=tf.nn.selu,  # Use SELU because it works better :)
                             name='dense_layer_1',
                             kernel_regularizer=regularizer,
                             bias_regularizer=regularizer)
        h2 = tf.layers.dense(h1,
                             100,
                             activation=None,
                             name='dense_layer_2',
                             kernel_regularizer=regularizer,
                             bias_regularizer=regularizer)
        h2_normalized = tf.nn.selu(tf.layers.dropout(h2, rate=0.1))
        h3 = tf.layers.dense(h2_normalized,
                             50,
                             activation=None,
                             name='dense_layer_3',
                             kernel_regularizer=regularizer,
                             bias_regularizer=regularizer)
        h3_normalized = tf.nn.selu(tf.layers.dropout(h3, rate=0.1))

        h4 = tf.layers.dense(h3_normalized,
                             1,
                             activation=None,
                             name='dense_layer_4',
                             kernel_regularizer=regularizer,
                             bias_regularizer=regularizer)
    else:
        assert False, "dom_class_type not found %s" % params.dom_class_type
    # Keep track of the logits to see if they don't explode
    tf.summary.histogram('Domain_logits', h4, family='domain_logits')
    return h4


def add_diff_noise(representations):
    """
    Add differentiable noise to the features before discriminator

    Resources:
      * http://www.inference.vc/instance-noise-a-trick-for-stabilising-gan-training/
      * https://arxiv.org/abs/1701.04862

    :param representations:
    :return:
    """
    global_step = tf.train.get_or_create_global_step()
    # Decrease the noise sigma during training
    sigma = 0.1 - 0.1*tf.clip_by_value(tf.cast(global_step, tf.float32) / 20000., 0.0, 1.0)
    tf.summary.scalar("sigma_add_diff", sigma, family='optimizer')  # Track the noise sigma in tensorboard
    return representations + sigma*tf.random_normal(representations.get_shape())
