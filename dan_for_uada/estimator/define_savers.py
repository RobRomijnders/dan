import tensorflow as tf
import logging
log = logging.getLogger('semantic_segmentation')


def train_saver(config, params, scope='saver'):
    del params, scope
    return tf.train.Saver(tf.global_variables(),
                          sharded=True,
                          max_to_keep=config.keep_checkpoint_max,
                          save_relative_paths=True)


def evaluate_saver(config, params, scope='saver'):
    """
    Additional scripting in the case we want to restore the EMA's
    :param config:
    :param params:
    :param scope:
    :return:
    """
    if params.restore_emas:
        log.debug('You are restroring the EMA weights')

    if params.ckpt_path is None:
        ckpt_path = tf.train.latest_checkpoint(params.log_dir)
    else:
        ckpt_path = params.ckpt_path
    log.debug(f'Initialize from checkpoint {ckpt_path}')

    with tf.name_scope(scope):
        log.debug('Do initialization for evaluation')

        ckpt_vars = tf.train.list_variables(ckpt_path)
        ckpt_var_names, ckpt_var_shapes = zip(*ckpt_vars)

        var_dict = {var.op.name: var for var in tf.global_variables()}

        found_init_ema = False
        if params.restore_emas:
            for var_name, var in var_dict.items():
                new_var_name = var_name + '/ExponentialMovingAverage'
                if new_var_name in ckpt_var_names:
                    var_dict.pop(var_name)
                    var_dict[new_var_name] = var

                    found_init_ema = True

        if found_init_ema:
            log.debug('Initialize the ExponentialMovingAverages from checkpoint')
        else:
            log.debug('Initialize the original variables from checkpoint')

        saver = tf.train.Saver(var_list=var_dict,
                               sharded=True,
                               max_to_keep=config.keep_checkpoint_max,
                               save_relative_paths=True)
    return saver


predict_saver = evaluate_saver

# with tf.name_scope(scope):
#     var_dict = dict()
#
#     for model_var in tf.model_variables():
#         model_var_name = model_var.op.name + ('/ExponentialMovingAverage' if params.restore_emas else '')
#         var_dict[model_var_name] = model_var
#
#         # Sanity check for the softmax output
#         if not has_printed and 'softmax' in model_var.op.name:
#             log.debug('The current model definition in code defines %s output classes' % str(model_var.shape[-1]))
#             has_printed = True
#
#     # for now only global_step is in rest_vars
#     rest_vars = set(tf.global_variables()).difference(set(var_dict.values()))
#     for rv in rest_vars:
#         var_dict[rv.op.name] = rv