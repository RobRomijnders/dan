"""Losses for Semantic Segmentation model supporting void labels.
"""

import tensorflow as tf
import logging
log = logging.getLogger('semantic_segmentation')
epsilon = 1E-10


def define_losses(mode, config, params, predictions, labels, domain_labels=None):
    """
    labels: Nb x hf x wf, tf.int32, with elements in [0,Nc-1]
    logits: Nb x hf x wf x Nc, tf.float32
    by convention, if void label exists it has index Nc-1 and is ignored in the losses
    """
    del config
    # --Segmentation loss--
    largest_cid = max(params.training_lids2cids)
    # ignore largest cid (void) if void class exists
    if -1 in params.training_problem_def['lids2cids']:
        largest_cid -= 1
    weights = tf.cast(labels <= largest_cid, tf.float32)

    if mode == tf.estimator.ModeKeys.TRAIN and len(params.tfrecords_list) > 1 \
            and params.unsupervised_adaptation:
        coefficients = [[[int(i == 0)]] for i, num in enumerate(params.Nb_list) for _ in range(num)]
        weights *= tf.constant(coefficients, dtype=tf.float32)
        log.debug('Unsupervised adaptation: The weights zero out all but the first %i label maps' % params.Nb_list[0])

    seg_loss = tf.losses.sparse_softmax_cross_entropy(
        labels,
        predictions['logits'],
        weights=weights,
        reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)

    # --Regularisation loss--
    reg_loss = tf.accumulate_n(tf.losses.get_regularization_losses())

    # --Domain loss--
    # Add the domain loss if we have more than one domain in our input pipeline
    if mode == tf.estimator.ModeKeys.TRAIN and len(params.tfrecords_list) > 1 and params.switch_train_op:
        log.debug('LOSSES: the domain loss is included')
        domain_loss = tf.losses.sigmoid_cross_entropy(domain_labels, logits=predictions['domain_logits'])
        domain_loss = tf.clip_by_value(domain_loss, clip_value_min=0.0, clip_value_max=5.0)
    else:
        log.debug('LOSSES: the domain loss is zero-ed out')
        domain_loss = 0.0

    # --Confusion loss--
    conf_weight = getattr(params, 'lambda_conf', 1.0)  # Weight the confusion loss by a coefficient
    if mode == tf.estimator.ModeKeys.TRAIN and len(params.tfrecords_list) > 1 and params.switch_train_op:
        domain_probs = tf.nn.sigmoid(predictions['domain_logits'])
        # domain_probs = tf.clip_by_value(domain_probs, epsilon, 1. - epsilon)
        if params.confusion_loss == 'cross':
            confusion_loss_unfiltered = -1 / 2 * tf.log(domain_probs * (1 - domain_probs))
            confusion_loss_unfiltered = tf.reduce_mean(tf.clip_by_value(confusion_loss_unfiltered,
                                                                        clip_value_min=0.0,
                                                                        clip_value_max=3.0))
            log.debug('LOSSES: you are using cross entropy confusion loss')
        else:
            confusion_loss_unfiltered = tf.cast(domain_labels, tf.float32)*(-1.*tf.log(1.-domain_probs + 1E-9))
            confusion_loss_unfiltered = tf.reduce_mean(tf.clip_by_value(confusion_loss_unfiltered,
                                                                        clip_value_min=0.0,
                                                                        clip_value_max=2.0))
            log.debug('LOSSES: you are using target confusion loss')

        # Zero out confusion loss for initial thousand steps
        global_step = tf.train.get_or_create_global_step()
        # confusion_loss = confusion_loss_unfiltered * tf.cast(tf.greater(global_step, 1000), tf.float32)
        a = getattr(params, 'ramp_start', 3000)
        if a > 0:
            start_step = params.num_batches_per_epoch * a * 2
            stop_step = params.num_batches_per_epoch * (a + 1) * 2

            ramp_coefficient = tf.clip_by_value(1. / (stop_step - start_step) * (tf.cast(global_step, tf.float32)
                                                                                 - start_step), 0.0, 1.0)
            log.debug(f'Ramp up the confusion loss lambda between {start_step} '
                      f'and {stop_step} to final weight {conf_weight}')
        else:
            start_step = -1 * a * params.num_batches_per_epoch * 2
            ramp_coefficient = tf.cast(tf.greater(global_step, start_step), tf.float32)
            log.debug(f'Start with confusion loss at step {start_step}, with weight {conf_weight}')
        tf.summary.scalar('ramp_coefficient', ramp_coefficient, family='optimizer')
        confusion_loss = confusion_loss_unfiltered * ramp_coefficient

    else:
        log.debug('LOSSES: The confusion loss is zero-ed out')
        confusion_loss = confusion_loss_unfiltered = 0.0

    # Combine all losses
    tot_loss = seg_loss + reg_loss + conf_weight*confusion_loss
    losses = {'total': tot_loss,
              'segmentation': seg_loss,
              'regularization': reg_loss,
              'confusion': confusion_loss_unfiltered}

    if mode == tf.estimator.ModeKeys.TRAIN:
        losses['domain'] = domain_loss

    if mode == tf.estimator.ModeKeys.TRAIN and params.save_summaries_steps < 5:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.name.replace(':', '__'), var, family='weights')
            tf.summary.scalar(var.name.replace(':', '__'), tf.reduce_max(var), family='max_values')
    return losses
