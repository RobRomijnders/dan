import tensorflow as tf


def mean_iou(labels, decisions, num_classes, params):
    """
    Calculates the mean IOU in tensorflow graph
    :param labels:
    :param decisions:
    :param num_classes:
    :param params:
    :return:
    """
    def do_flatten(tensor):
        return tf.reshape(tensor, [-1])

    conf_matrix = tf.confusion_matrix(labels=do_flatten(labels),
                                      predictions=do_flatten(decisions),
                                      num_classes=num_classes)
    if -1 in params.training_problem_def['lids2cids']:
        conf_matrix = conf_matrix[:-1, :-1]
    inter = tf.diag_part(conf_matrix)
    union = tf.reduce_sum(conf_matrix, 0) + tf.reduce_sum(conf_matrix, 1) - inter

    inter = tf.cast(inter, tf.float32)
    union = tf.cast(union, tf.float32) + 1E-9
    m_iou = tf.reduce_mean(tf.div(inter, union))
    return m_iou


def accuracy(labels, logits):
    """
    Calculates the accuracy in tensorflow graph.

    Assumes a binary problem
    :param labels:
    :param logits:
    :return:
    """
    with tf.control_dependencies([tf.assert_rank(logits, 1), tf.assert_rank(labels, 1)]):
        pred_labels = tf.cast(tf.greater(logits, 0.0), dtype=tf.int32)
        acc = tf.reduce_mean(tf.cast(tf.equal(labels, pred_labels), dtype=tf.float32))
    return acc
