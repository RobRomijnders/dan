from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.layers import utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import math_ops
from tensorflow import concat, add_to_collection, assign, stop_gradient
from tensorflow.python.layers.normalization import BatchNormalization as BatchNormalization_source
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops

import logging

log = logging.getLogger('semantic_segmentation')


class BatchNormalizationCustom(BatchNormalization_source):
    def __init__(self, Nb_list=None, *args, **kwargs):
        super(BatchNormalizationCustom, self).__init__(*args, **kwargs)
        self.Nb_list = Nb_list

    def assign_instant_statistics(self, mean, variance):
        """
        adds an op to the graph for instantly assigning the current mean to the moving mean
        This is used in adaptation to prevent a cold start
        :param mean:
        :param variance:
        :return:
        """
        instant_assign_mean_op = assign(self.moving_mean, mean, name='instant_assign_mean')
        instant_assign_variance_op = assign(self.moving_variance, variance, name='instant_assign_variance')
        add_to_collection('instant_assignments', instant_assign_mean_op)
        add_to_collection('instant_assignments', instant_assign_variance_op)

    def _assign_moving_average(self, variable, value, one_minus_decay):
        with ops.name_scope(None, 'AssignMovingAvg',
                            [variable, value, one_minus_decay]) as scope:
            with ops.colocate_with(variable):
                update_delta = math_ops.multiply(
                    math_ops.subtract(variable.read_value(), value),
                    one_minus_decay)
                if isinstance(variable, resource_variable_ops.ResourceVariable):
                    # state_ops.assign_sub does an extra read_variable_op after the
                    # assign. We avoid that here.
                    return gen_resource_variable_ops.assign_sub_variable_op(
                        variable.handle, update_delta, name=scope)
                else:
                    return state_ops.assign_sub(variable, update_delta, name=scope)

    def _fused_batch_norm(self, inputs, training):
        """Returns the output of fused batch norm."""
        # TODO(reedwm): Add support for fp16 inputs.
        beta = self.beta if self.center else self._beta_const
        gamma = self.gamma if self.scale else self._gamma_const

        def _fused_batch_norm_training():
            return nn.fused_batch_norm(
                inputs[:self.Nb_list[0]],
                gamma,
                beta,
                epsilon=self.epsilon,
                data_format=self._data_format)

        def _fused_batch_norm_inference():
            return nn.fused_batch_norm(
                inputs,
                gamma,
                beta,
                mean=self.moving_mean,
                variance=self.moving_variance,
                epsilon=self.epsilon,
                is_training=False,
                data_format=self._data_format)

        output, mean, variance = utils.smart_cond(
            training, _fused_batch_norm_training, _fused_batch_norm_inference)
        add_to_collection('instant_means', mean)
        add_to_collection('instant_variances', variance)
        add_to_collection('moving', self.moving_mean)
        add_to_collection('moving', self.moving_variance)
        add_to_collection('bn', mean)
        add_to_collection('bn', variance)
        self.assign_instant_statistics(mean, variance)

        if training and len(self.Nb_list) > 1:
            # TODO (rob) use fused batch norm here. it is 6X faster
            # https://github.com/tensorflow/tensorflow/issues/7551#issuecomment-280421351
            inv = math_ops.rsqrt(stop_gradient(variance) + self.epsilon) * gamma
            second_output = inputs[self.Nb_list[0]:] * inv + beta - stop_gradient(mean) * inv
            log.debug('You are doing custom batchnorm by overwriting the BatchNormClass (1)')
            output = concat((output, second_output), axis=0)

        if not self._bessels_correction_test_only:
            # Remove Bessel's correction to be consistent with non-fused batch norm.
            # Note that the variance computed by fused batch norm is
            # with Bessel's correction.
            sample_size = math_ops.cast(
                array_ops.size(inputs) / array_ops.size(variance), variance.dtype)
            factor = (sample_size - math_ops.cast(1.0, variance.dtype)) / sample_size
            variance *= factor

        training_value = utils.constant_value(training)
        if training_value is None:
            one_minus_decay = utils.smart_cond(training,
                                               lambda: 1.0 - self.momentum,
                                               lambda: 0.)
        else:
            one_minus_decay = ops.convert_to_tensor(1.0 - self.momentum)
        if training_value or training_value is None:
            mean_update = self._assign_moving_average(self.moving_mean, mean,
                                                      one_minus_decay)
            variance_update = self._assign_moving_average(self.moving_variance,
                                                          variance, one_minus_decay)
            if True:
                # Note that in Eager mode, the updates are already executed when running
                # assign_moving_averages. So we do not need to put them into
                # collections.
                self.add_update(mean_update, inputs=inputs)
                self.add_update(variance_update, inputs=inputs)

        return output
