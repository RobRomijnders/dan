from tensorflow.contrib.framework.python.ops import add_arg_scope
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.python.framework import ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import variables as tf_variables
from utils.utils_bn_class_1 import BatchNormalizationCustom
import logging
log = logging.getLogger('semantic_segmentation')

DATA_FORMAT_NCHW = 'NCHW'
DATA_FORMAT_NHWC = 'NHWC'
DATA_FORMAT_NCDHW = 'NCDHW'
DATA_FORMAT_NDHWC = 'NDHWC'


@add_arg_scope
def batch_norm(inputs,
               decay=0.999,
               center=True,
               scale=False,
               epsilon=0.001,
               activation_fn=None,
               param_initializers=None,
               param_regularizers=None,
               updates_collections=ops.GraphKeys.UPDATE_OPS,
               is_training=True,
               reuse=None,
               variables_collections=None,
               outputs_collections=None,
               trainable=True,
               batch_weights=None,
               fused=None,
               data_format=DATA_FORMAT_NHWC,
               zero_debias_moving_mean=False,
               scope=None,
               renorm=False,
               renorm_clipping=None,
               renorm_decay=0.99,
               Nb_list=None):
    """Adds a Batch Normalization layer from http://arxiv.org/abs/1502.03167.

      "Batch Normalization: Accelerating Deep Network Training by Reducing
      Internal Covariate Shift"

      Sergey Ioffe, Christian Szegedy

    Can be used as a normalizer function for conv2d and fully_connected.

    Note: when training, the moving_mean and moving_variance need to be updated.
    By default the update ops are placed in `tf.GraphKeys.UPDATE_OPS`, so they
    need to be added as a dependency to the `train_op`. For example:

    ```python
      update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
      with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss)
    ```

    One can set updates_collections=None to force the updates in place, but that
    can have a speed penalty, especially in distributed settings.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`. The normalization is over all but the last dimension if
        `data_format` is `NHWC` and the second dimension if `data_format` is
        `NCHW`.
      decay: Decay for the moving average. Reasonable values for `decay` are close
        to 1.0, typically in the multiple-nines range: 0.999, 0.99, 0.9, etc.
        Lower `decay` value (recommend trying `decay`=0.9) if model experiences
        reasonably good training performance but poor validation and/or test
        performance. Try zero_debias_moving_mean=True for improved stability.
      center: If True, add offset of `beta` to normalized tensor. If False, `beta`
        is ignored.
      scale: If True, multiply by `gamma`. If False, `gamma` is
        not used. When the next layer is linear (also e.g. `nn.relu`), this can be
        disabled since the scaling can be done by the next layer.
      epsilon: Small float added to variance to avoid dividing by zero.
      activation_fn: Activation function, default set to None to skip it and
        maintain a linear activation.
      param_initializers: Optional initializers for beta, gamma, moving mean and
        moving variance.
      param_regularizers: Optional regularizer for beta and gamma.
      updates_collections: Collections to collect the update ops for computation.
        The updates_ops need to be executed with the train_op.
        If None, a control dependency would be added to make sure the updates are
        computed in place.
      is_training: Whether or not the layer is in training mode. In training mode
        it would accumulate the statistics of the moments into `moving_mean` and
        `moving_variance` using an exponential moving average with the given
        `decay`. When it is not in training mode then it would use the values of
        the `moving_mean` and the `moving_variance`.
      reuse: Whether or not the layer and its variables should be reused. To be
        able to reuse the layer scope must be given.
      variables_collections: Optional collections for the variables.
      outputs_collections: Collections to add the outputs.
      trainable: If `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      batch_weights: An optional tensor of shape `[batch_size]`,
        containing a frequency weight for each batch item. If present,
        then the batch normalization uses weighted mean and
        variance. (This can be used to correct for bias in training
        example selection.)
      fused: if `True`, use a faster, fused implementation if possible.
        If `None`, use the system recommended implementation.
      data_format: A string. `NHWC` (default) and `NCHW` are supported.
      zero_debias_moving_mean: Use zero_debias for moving_mean. It creates a new
        pair of variables 'moving_mean/biased' and 'moving_mean/local_step'.
      scope: Optional scope for `variable_scope`.
      renorm: Whether to use Batch Renormalization
        (https://arxiv.org/abs/1702.03275). This adds extra variables during
        training. The inference is the same for either value of this parameter.
      renorm_clipping: A dictionary that may map keys 'rmax', 'rmin', 'dmax' to
        scalar `Tensors` used to clip the renorm correction. The correction
        `(r, d)` is used as `corrected_value = normalized_value * r + d`, with
        `r` clipped to [rmin, rmax], and `d` to [-dmax, dmax]. Missing rmax, rmin,
        dmax are set to inf, 0, inf, respectively.
      renorm_decay: Momentum used to update the moving means and standard
        deviations with renorm. Unlike `momentum`, this affects training
        and should be neither too small (which would add noise) nor too large
        (which would give stale estimates). Note that `decay` is still applied
        to get the means and variances for inference.

    Returns:
      A `Tensor` representing the output of the operation.

    Raises:
      ValueError: If `data_format` is neither `NHWC` nor `NCHW`.
      ValueError: If the rank of `inputs` is undefined.
      ValueError: If rank or channels dimension of `inputs` is undefined.
    """
    if fused is None:
        fused = True

    # Only use _fused_batch_norm if all of the following three
    # conditions are true:
    # (1) fused is set True;
    # (2) it is possible to use (currently it doesn't support batch weights,
    #   renorm, and the case when rank is neither 2 nor 4);
    # (3) it is used with zero_debias_moving_mean, or an input shape of rank 2,
    #   or non-default updates_collections (not implemented in
    #   normalization_layers.BatchNormalization yet); otherwise use the fused
    #   implementation in normalization_layers.BatchNormalization.
    inputs = ops.convert_to_tensor(inputs)
    # rank = inputs.get_shape().ndims

    if data_format not in (DATA_FORMAT_NCHW, DATA_FORMAT_NHWC):
        raise ValueError('data_format has to be either NCHW or NHWC.')

    layer_variable_getter = _build_variable_getter()
    with variable_scope.variable_scope(scope, 'BatchNorm', [inputs], reuse=reuse,
                                       custom_getter=layer_variable_getter) as sc:
        inputs = ops.convert_to_tensor(inputs)

        # Determine whether we can use the core layer class.
        if batch_weights is None and updates_collections is ops.GraphKeys.UPDATE_OPS and not zero_debias_moving_mean:
            # Use the core layer class.
            axis = 1 if data_format == DATA_FORMAT_NCHW else -1
            if not param_initializers:
                param_initializers = {}
            beta_initializer = param_initializers.get('beta',
                                                      init_ops.zeros_initializer())
            gamma_initializer = param_initializers.get('gamma',
                                                       init_ops.ones_initializer())
            moving_mean_initializer = param_initializers.get(
                'moving_mean', init_ops.zeros_initializer())
            moving_variance_initializer = param_initializers.get(
                'moving_variance', init_ops.ones_initializer())
            if not param_regularizers:
                param_regularizers = {}
            beta_regularizer = param_regularizers.get('beta')
            gamma_regularizer = param_regularizers.get('gamma')
            layer = BatchNormalizationCustom(
                axis=axis,
                momentum=decay,
                epsilon=epsilon,
                center=center,
                scale=scale,
                beta_initializer=beta_initializer,
                gamma_initializer=gamma_initializer,
                moving_mean_initializer=moving_mean_initializer,
                moving_variance_initializer=moving_variance_initializer,
                beta_regularizer=beta_regularizer,
                gamma_regularizer=gamma_regularizer,
                trainable=trainable,
                renorm=renorm,
                renorm_clipping=renorm_clipping,
                renorm_momentum=renorm_decay,
                name=sc.name,
                _scope=sc,
                _reuse=reuse,
                fused=fused,
                Nb_list=Nb_list)
            outputs = layer.apply(inputs, training=is_training)

            # Add variables to collections.
            _add_variable_to_collections(
                layer.moving_mean, variables_collections, 'moving_mean')
            _add_variable_to_collections(
                layer.moving_variance, variables_collections, 'moving_variance')
            if layer.beta is not None:
                _add_variable_to_collections(layer.beta, variables_collections, 'beta')
            if layer.gamma is not None:
                _add_variable_to_collections(
                    layer.gamma, variables_collections, 'gamma')

            if activation_fn is not None:
                outputs = activation_fn(outputs)
            return utils.collect_named_outputs(outputs_collections, sc.name, outputs)


def _model_variable_getter(getter, name, shape=None, dtype=None,
                           initializer=None, regularizer=None, trainable=True,
                           collections=None, caching_device=None,
                           partitioner=None, rename=None, use_resource=None,
                           **_):
    """Getter that uses model_variable for compatibility with core layers."""
    short_name = name.split('/')[-1]
    if rename and short_name in rename:
        name_components = name.split('/')
        name_components[-1] = rename[short_name]
        name = '/'.join(name_components)
    return variables.model_variable(
        name, shape=shape, dtype=dtype, initializer=initializer,
        regularizer=regularizer, collections=collections, trainable=trainable,
        caching_device=caching_device, partitioner=partitioner,
        custom_getter=getter, use_resource=use_resource)


def _build_variable_getter(rename=None):
    """Build a model variable getter that respects scope getter and renames."""

    # VariableScope will nest the getters
    def layer_variable_getter(getter, *args, **kwargs):
        kwargs['rename'] = rename
        return _model_variable_getter(getter, *args, **kwargs)

    return layer_variable_getter


def _add_variable_to_collections(variable, collections_set, collections_name):
    """Adds variable (or all its parts) to all collections with that name."""
    collections = utils.get_variable_collections(
        collections_set, collections_name) or []
    variables_list = [variable]
    if isinstance(variable, tf_variables.PartitionedVariable):
        variables_list = [v for v in variable]
    for collection in collections:
        for var in variables_list:
            if var not in ops.get_collection(collection):
                ops.add_to_collection(collection, var)
