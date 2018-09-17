from tensorflow.contrib.training import create_train_op
import tensorflow as tf
import logging
log = logging.getLogger('semantic_segmentation')
from tensorflow import clip_by_global_norm, clip_by_value


def create_alternating_train_op(losses, optimizer, global_step, params):
    """
    This op control the alternating of training ops

    Every `switch_period`, another train_op will be used with its respective trainable variables

    :param losses: Dictionary of losses in the network
    :param optimizer: optimizer to supply to `create_train_op`
    :param global_step: global step tensor
    :param params:
    :return:
    """
    def clip_function(gv_list):
        if params.gradient_clip_norm > 0.0:
            grad_list, var_list = zip(*gv_list)

            def clipping(x):
                return clip_by_value(x,
                                     clip_value_min=-params.gradient_clip_norm,
                                     clip_value_max=params.gradient_clip_norm)
            grad_list = map(clipping, grad_list)
            # grad_list, _ = clip_by_global_norm(list(grad_list), clip_norm=params.gradient_clip_norm)
            return zip(grad_list, var_list)
        else:
            return gv_list

    # We only alternate the training op if it is set to True AND when there's actually two domains
    switch_train_op = params.switch_train_op and len(params.tfrecords_list) > 1

    if switch_train_op:
        switch_period = params.switch_period

        variables_sem_seg = tf.trainable_variables()
        variables_dom_class = tf.trainable_variables('domain_classifier')
        for var in variables_dom_class:
            variables_sem_seg.remove(var)

        log.debug('We are switching the train ops between Sem Seg %i tensors and Dom Class %i tensors' %
                  (len(variables_sem_seg), len(variables_dom_class)))
        condition = tf.greater_equal(tf.mod(global_step, 2*switch_period), switch_period)
        train_op = tf.cond(condition,
                           true_fn=lambda: create_train_op(
                                losses['total'],
                                optimizer,
                                variables_to_train=variables_sem_seg,
                                global_step=global_step,
                                check_numerics=False,
                                transform_grads_fn=clip_function),
                           false_fn=lambda: create_train_op(
                                losses['domain'],
                                optimizer,
                                variables_to_train=variables_dom_class,
                                global_step=global_step,
                                check_numerics=False,
                                transform_grads_fn=clip_function))
        tf.summary.scalar('Switch_condition', tf.cast(condition, tf.int16), family='optimizer')
    else:
        train_op = create_train_op(
            losses['total'],
            optimizer,
            global_step=global_step,
            check_numerics=False,)
    return train_op


def count_trainable_vars():
    """
    Counts the number of trainable variables in your tensorflow model
    :return:
    """
    # add a training hook if you wanted to know how many trainable variables we have
    #         tf.train.LoggingTensorHook([count_trainable_vars()], every_n_iter=1)
    trainable_vars = tf.trainable_variables()

    num_elements = tf.zeros([], dtype=tf.int32)
    for var in trainable_vars:
        num_elements += tf.reduce_prod(tf.shape(var))
    return tf.identity(num_elements, name='num_elements')
