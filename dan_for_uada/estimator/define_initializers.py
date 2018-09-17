import os
import tensorflow as tf
import logging
log = logging.getLogger('semantic_segmentation')


def train_init(config, params, scope='init'):
    # different situations for initialization:
    #   1) initialize from init_ckpt_path (log_dir has to be empty from checkpoints)
    #   2) continue training from log_dir

    del config

    with tf.name_scope(scope), tf.device('/cpu:0'):

        # one of those must given (xor)
        # assert bool(params.init_ckpt_path) != bool(tf.train.latest_checkpoint(params.log_dir)), (
        #     'If init_ckpt_path is given log_dir has to be empty of checkpoints, '
        #     'if log_dir is given training continuous from latest checkpoint and '
        #     'init_ckpt_path has to be empty.')

        # -- initialize from checkpoint, e.g. trained on ImageNet
        if params.init_ckpt_path not in ['', None]:
            # the best we can do is to initialize from the checkpoint as much variables as possible
            # so we find the mapping from checkpoint names to model names
            # assumes names in model are extended with a prefix from names in checkpoint
            # e.g.
            # in checkpoint: resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights
            # in model: feature_extractor/base/resnet_v1_50/block1/unit_1/bottleneck_v1/conv1/weights
            log.debug(f'Initialize parameters from {params.init_ckpt_path}')

            # list of (name, shape) of checkpoint variables
            ckpt_vars = tf.train.list_variables(params.init_ckpt_path)
            # list of tf.Variable of model variables
            global_vars = tf.global_variables()

            extra_vars_to_init = set(global_vars)

            # checkpoint variable name --> model variable mappings
            # TODO: exclude variables in a better way, still the parts below may be included in a
            # useful variable, e.g. use moving_average_variables and variables from model.py
            exclude_vars = ['global_step', 'train_ops', 'ExponentialMovingAverage',
                            'Momentum', 'domain_classifier']  # 'classifier', 'extension'
            var_dict = dict()
            var_dict_bn = dict()  # Additional dictionary to save the mappings for when splitting the batchnorm layer
            for global_var in global_vars:
                for exclude_var in exclude_vars:
                    if exclude_var in global_var.op.name:
                        break
                else:
                    for var_name, var_shape in ckpt_vars:
                        if var_name in global_var.op.name.replace('bn_split/', '') and tf.TensorShape(var_shape).is_compatible_with(global_var.shape):
                            var_dict[var_name] = [global_var]
                            extra_vars_to_init.remove(global_var)
                            log.debug(f'Initialize {var_name:100s} from checkpoint')
                        elif var_name in global_var.op.name.replace('split_bn_target/', '') and tf.TensorShape(var_shape).is_compatible_with(global_var.shape):
                            var_dict_bn[var_name] = global_var
                            extra_vars_to_init.remove(global_var)
                            log.debug(f'Initialize {var_name:100s} from checkpoint onto the bn target')

            # Add the variables of the splitted batch norm layers also to the var_dict
            for var_name, global_var in var_dict_bn.items():
                var_dict[var_name].append(global_var)

            # Initialize the possible tensors from the checkpoint
            init_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(
                params.init_ckpt_path,
                var_dict,
                ignore_missing_vars=False)

            # Initialize the other variables with their individual initializers
            extra_init_op = tf.variables_initializer(extra_vars_to_init)
            return tf.group(init_op, extra_init_op), init_feed_dict
        else:
            return None, None