import os
import itertools
from operator import itemgetter
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tensorflow.python.client import timeline
from tensorflow.python.ops import metrics_impl

from estimator.define_losses import define_losses
from estimator.define_savers import train_saver, predict_saver, evaluate_saver
from estimator.define_initializers import train_init
from estimator.define_custom_metrics import mean_iou, accuracy
from estimator.util_estimator import create_alternating_train_op
from utils.util_hooks import CheckpointSaverHookCustom

import logging


_ALLOWED_MODES = {tf.estimator.ModeKeys.TRAIN,
                  tf.estimator.ModeKeys.EVAL,
                  tf.estimator.ModeKeys.PREDICT}

log = logging.getLogger('semantic_segmentation')


def define_estimator(mode, features, labels, model_fn, config, params):
    """Add documentation... More information at tf.Estimator class.
    Assumptions:
      features: a dict containing rawfeatures and profeatures
        both: Nb x hf x wf x 3, tf.float32, in [0,1]
      labels: a dict containing rawfeatures and profeatures
        both: Nb x hf x wf, tf.int32, in [0,Nc-1]
    Args:
      features: First item returned by input_fn passed to train, evaluate, and predict.
      labels: Second item returned by input_fn passed to train, evaluate, and predict.
      mode: one of tf.estimator.ModeKeys.
      model_fn: the model function that maps images to predictions
      config: a tf.estimator.RunConfig object...
      params: a tf.train.HParams object...
      ...
    """
    assert mode in _ALLOWED_MODES, (
        'mode should be TRAIN, EVAL or PREDICT from tf.estimator.ModeKeys.')
    assert params.name_feature_extractor in {'resnet_v1_50', 'resnet_v1_101'}, (
        'params must have name_feature_extractor attribute in resnet_v1_{50,101}.')

    # unpack features
    proimages = features['proimages']
    prolabels = labels['prolabels'] if mode != tf.estimator.ModeKeys.PREDICT else None

    # -- build a fully convolutional model for semantic segmentation
    _, _, predictions = model_fn(mode, proimages, prolabels, config, params)

    # create training ops and exponential moving averages
    if mode == tf.estimator.ModeKeys.TRAIN:
        if params.switch_train_op:
            with tf.variable_scope('domain_labels'):
                # Manually create the domain labels
                # TODO (rob) work around these domain labels with custom domain loss ??
                num_feature_vecs = int(params.height_feature_extractor *
                                       params.width_feature_extractor / params.stride_feature_extractor ** 2)
                domain_labels_tiled = tf.tile(tf.expand_dims(labels['domainlabels'], 1), [1, num_feature_vecs])
                domain_labels = tf.reshape(domain_labels_tiled, [-1])
                labels['domainlabels'] = domain_labels
        else:
            domain_labels = None

        # global step
        global_step = tf.train.get_or_create_global_step()

        # losses
        with tf.variable_scope('losses'):
            losses = define_losses(mode, config, params, predictions, prolabels, domain_labels)

        # exponential moving averages
        # creates variables in checkpoint with name: 'emas/' + <variable_name> +
        #   {'ExponentialMovingAverage,Momentum}
        # ex.: for 'classifier/logits/Conv/biases' it saves also
        #          'emas/classifier/logits/Conv/biases/ExponentialMovingAverage'
        #      and 'emas/classifier/logits/Conv/biases/Momentum'
        # create_train_op guarantees to run GraphKeys.UPDATE_OPS collection
        #   before total_loss in every step, but doesn't give any guarantee
        #   for running after some other op, and since ema need to be run
        #   after applying the gradients maybe this code needs checking
        if params.ema_decay > 0:
            with tf.name_scope('exponential_moving_averages'):
                # for mv in slim.get_model_variables():
                #  log.debug('slim.model_vars:', mv.op.name)
                log.debug('Record exponential weighted moving averages')
                ema = tf.train.ExponentialMovingAverage(params.ema_decay,
                                                        num_updates=global_step,
                                                        zero_debias=True)
                maintain_ema_op = ema.apply(var_list=list(filter(lambda x: 'domain_classifier' not in x.name,
                                                                 tf.trainable_variables())))
                tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, maintain_ema_op)

        # create training operation
        with tf.variable_scope('train_ops'):
            learning_rate = tf.train.piecewise_constant(global_step,
                                                        params.lr_boundaries,
                                                        params.lr_values)
            # optimizer
            if params.optimizer == 'SGDM':
                optimizer = tf.train.MomentumOptimizer(
                    learning_rate,
                    params.momentum,
                    use_nesterov=params.use_nesterov)
            elif params.optimizer == 'SGD':
                optimizer = tf.train.GradientDescentOptimizer(learning_rate)
            else:
                log.warning('Optimizer type not found (%s)' % params.optimizer)
                optimizer = None
            # training op
            train_op = create_alternating_train_op(losses, optimizer, global_step, params)

        training_hooks = []
            # _RunMetadataHook(params.log_dir,
            #                  every_n_iter=max(params.num_training_steps // 50,
            #                                   params.save_checkpoints_steps))]

        summaries_data = {'features': features,
                          'labels': labels,
                          'predictions': predictions,
                          'losses': losses,
                          'learning_rate': learning_rate}

    if mode == tf.estimator.ModeKeys.EVAL:
        with tf.variable_scope('losses'):
            losses = define_losses(mode, config, params, predictions, prolabels)

        # returns (variable, update_op)
        # TF internal error/problem: _streaming_confusion_matrix internally casts
        # labels and predictions to int64, and since we feed a dictionary, tensors are
        # passed by reference leading them to change type, thus we send an identity
        # confusion_matrix = metrics_impl._streaming_confusion_matrix(  # pylint: disable=protected-access
        #     tf.identity(prolabels),
        #     tf.identity(predictions['decisions']),
        #     params.training_Nclasses)
        confusion_matrix = metrics_impl._streaming_confusion_matrix(  # pylint: disable=protected-access
            prolabels,
            predictions['decisions'],
            params.training_Nclasses)

        # dict of metrics keyed by name with values tuples of (metric_tensor, update_op)
        eval_metric_ops = {'confusion_matrix': (
            tf.to_int32(confusion_matrix[0]), confusion_matrix[1])}

    # -- create EstimatorSpec according to mode
    if mode == tf.estimator.ModeKeys.TRAIN:
        scaffold = _define_scaffold(mode, config, params, summaries_data)
        # training_hooks.append(CheckpointSaverHookCustom(start_step=int(params.num_batches_per_epoch * (params.Ne - 1)),
        #                                                 checkpoint_dir=params.log_dir,
        #                                                 scaffold=scaffold,
        #                                                 save_steps=int(params.num_batches_per_epoch / 5)))
        estimator_spec = tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            loss=losses['total'],
            train_op=train_op,
            training_hooks=training_hooks,
            scaffold=scaffold)
    elif mode == tf.estimator.ModeKeys.EVAL:
        scaffold = _define_scaffold(mode, config, params)
        estimator_spec = tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            loss=losses['total'],
            eval_metric_ops=eval_metric_ops,
            scaffold=scaffold)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        scaffold = _define_scaffold(mode, config, params)
        # workaround for connecting input pipeline outputs to system output
        # TODO: make it more clear
        predictions['rawimages'] = features['rawimages']
        predictions['rawimagespaths'] = features['rawimagespaths']
        # the expected predictions.keys() in this point is:
        # dict_keys(['logits', 'probabilities', 'decisions', 'rawimages', 'rawimagespaths'])
        estimator_spec = tf.estimator.EstimatorSpec(
            mode,
            predictions=predictions,
            scaffold=scaffold)
    else:
        assert False, f'No such mode {mode}'

    return estimator_spec


def _define_scaffold(mode, config, params, summaries_data=None):
    """Creates scaffold containing initializers, savers and summaries.

    Args:
      summaries_data: dictionary containing all tensors needed for summaries during training

    Returns:
      a tf.train.Scaffold instance
    """
    # Comment: init_op with init_feed_dict, and init_fn are executed from SessionManager
    # only if model is not loaded successfully from checkpoint using the saver.
    # if no saver is provided then the default saver is constructed to load all
    # variables (from collections GLOBAL_VARIABLES and SAVEABLE_OBJECTS) and init_op won't
    # be executed.
    # For that reason, during training using init_checkpoint we provide a custom saver only
    # for model variables and an init_op to initialize all variables not in init_checkpoint.

    # create scopes outside of scaffold namescope
    with tf.name_scope('init') as init_scope:
        pass
    with tf.name_scope('saver') as saver_scope:
        pass

    with tf.name_scope('scaffold'):
        if mode == tf.estimator.ModeKeys.TRAIN:
            _define_summaries(mode, config, params, summaries_data)
            saver = train_saver(config, params, scope=saver_scope)
            init_op, init_feed_dict = train_init(config, params, scope=init_scope)
        elif mode == tf.estimator.ModeKeys.EVAL:
            saver = evaluate_saver(config, params, scope=saver_scope)
            init_op, init_feed_dict = [None] * 2
        elif mode == tf.estimator.ModeKeys.PREDICT:
            saver = predict_saver(config, params, scope=saver_scope)
            init_op, init_feed_dict = [None] * 2
        else:
            assert False, "No such mode"

        # WARNING: default ready_op and ready_for_local_init_op install operations
        #   in the graph to report_uninitialized_variables, resulting in too many ops,
        #   so make ready_for_local_init_op a no_op to reduce them.
        scaffold = tf.train.Scaffold(
            init_op=init_op,
            init_feed_dict=init_feed_dict,
            saver=saver)

    return scaffold


def _define_summaries(mode, config, params, summaries_data):
    del config
    assert mode == tf.estimator.ModeKeys.TRAIN, log.debug('internal error: summaries only for training.')

    with tf.name_scope('summaries'), tf.device('/cpu:0'):
        # unpack necessary objects and tensors
        # WARNING: assumes all necessary items exist (maybe add assertions)
        rawimages = summaries_data['features']['rawimages']
        rawlabels = summaries_data['labels']['rawlabels']
        proimages = summaries_data['features']['proimages']
        prolabels = summaries_data['labels']['prolabels']
        _, probs, decs = itemgetter('logits', 'probabilities', 'decisions')(
            summaries_data['predictions'])
        tot_loss, reg_loss, seg_loss, domain_loss, confusion_loss = itemgetter('total', 'regularization',
                                                                               'segmentation', 'domain', 'confusion')(
            summaries_data['losses'])

        # drawing
        with tf.name_scope('drawing'):
            with tf.name_scope('palette'):
                palette = tf.constant(params.training_problem_def['cids2colors'], dtype=tf.uint8)

            # WARNING: assuming upsampling, that is all color_* images have the
            # same spatial dimensions
            color_decisions = _cids2col(decs, palette)
            # generate confidence image, preventing TF from normalizing max prob
            # to 1, by casting to tf.uint8
            color_confidences = tf.stack([tf.cast(tf.reduce_max(probs, axis=3) * 255, tf.uint8)] * 3, axis=3)

            # create a 2-by-2 collage of ground truth and results or separate images
            # sometimes due to different input preprocessing pro* and color_*
            # don't have the same dimensions
            # if (params.collage_image_summaries and
            #     _have_equal_shapes([color_labels, color_decisions, proimages, color_confidences])):
            #     # _have_compatible_shapes([color_labels, color_decisions, rawimages, color_confidences])):
            #   collage = tf.concat([tf.concat([color_labels, color_decisions], 1),
            #                        tf.concat([tf.image.convert_image_dtype(proimages, tf.uint8),
            #                                   color_confidences],
            #                                  1)],
            #                       2)
            #   tf.summary.image('results', collage, max_outputs=params.Nb, family='results')
            # else:
            tf.summary.image('rawimages', tf.image.convert_image_dtype(rawimages, tf.uint8), max_outputs=params.Nb,
                             family='raw_data')
            tf.summary.image('rawlabels', _cids2col(rawlabels, palette), max_outputs=params.Nb, family='raw_data')
            tf.summary.image('proimages', tf.image.convert_image_dtype(proimages, tf.uint8, saturate=True),
                             max_outputs=params.Nb, family='preprocessed_data')
            tf.summary.image('prolabels', _cids2col(prolabels, palette), max_outputs=params.Nb,
                             family='preprocessed_data')
            tf.summary.image('decisions', color_decisions, max_outputs=params.Nb, family='results')
            tf.summary.image('confidences', color_confidences, max_outputs=params.Nb, family='results')

            # compute batch metrics
            if len(params.tfrecords_list) == 1:
                m_iou = mean_iou(prolabels, decs, num_classes=params.training_Nclasses, params=params)
                tf.summary.scalar('mean_IOU_source', m_iou, family='performance')
            elif len(params.tfrecords_list) == 2:
                nb1 = params.Nb_list[0]
                m_iou_source = mean_iou(prolabels[:nb1], decs[:nb1],
                                        num_classes=params.training_Nclasses, params=params)
                tf.summary.scalar('mean_IOU_source', m_iou_source, family='performance')
                m_iou_target = mean_iou(prolabels[nb1:], decs[nb1:],
                                        num_classes=params.training_Nclasses, params=params)
                tf.summary.scalar('mean_IOU_target', m_iou_target, family='performance')

        # TODO: in order to disable loss summary created internally by estimator this line should
        # evaluate to False: not any([x.op.name == 'loss' for x in ops.get_collection(ops.GraphKeys.SUMMARIES)])

        # evaluate to False:
        # not any([x.op.name == 'loss' for x in ops.get_collection(ops.GraphKeys.SUMMARIES)])
        tf.summary.scalar('total', tot_loss, family='losses')
        tf.summary.scalar('regularization', reg_loss, family='losses')
        tf.summary.scalar('segmentation', seg_loss, family='losses')
        tf.summary.scalar('domain', domain_loss, family='losses')
        tf.summary.scalar('confusion', confusion_loss, family='losses')

        if params.switch_train_op:
            acc = accuracy(summaries_data['labels']['domainlabels'], summaries_data['predictions']['domain_logits'])
            tf.summary.scalar('domain_accuracy', acc, family='performance')

        tf.summary.scalar('learning_rate', summaries_data['learning_rate'], family='optimizer')


def _cids2col(cids, palette):
    # cids: Nb x H x W, tf.int32, with class ids in [0,Nc-1]
    # palette: Nc x 3, tf.uint8, with rgb colors in [0,255]
    # returns: Nb x H x W x 3, tf.uint8, in [0,255]

    # TODO: add type checking
    return tf.gather_nd(palette, tf.expand_dims(cids, axis=-1))


class _RunMetadataHook(tf.train.SessionRunHook):
    """Exports the run metadata as a trace to log_dir every N local steps or every N seconds.
    """

    # TODO: implement this with tf.profiler

    def __init__(self, log_dir, every_n_iter=None, every_n_secs=None):
        """Initializes a `_RunMetadataHook`.

        Args:
          log_dir: the log_dir directory to save traces.
          every_n_iter: `int`, save traces once every N local steps.
          every_n_secs: `int` or `float`, save traces once every N seconds.

          Exactly one of `every_n_iter` and `every_n_secs` should be provided.

        Raises:
          ValueError: if `every_n_iter` is non-positive.
        """
        if (every_n_iter is None) == (every_n_secs is None):
            raise ValueError("Exactly one of every_n_iter and every_n_secs must be provided.")
        if every_n_iter is not None and every_n_iter <= 0:
            raise ValueError(f"Invalid every_n_iter={every_n_iter}.")
        self._timer = tf.train.SecondOrStepTimer(every_secs=every_n_secs, every_steps=every_n_iter)
        self._iter_count = None
        self._should_trigger = None
        self._tf_global_step = None
        self._np_global_step = None
        self._log_dir = log_dir

    def begin(self):
        self._timer.reset()
        self._iter_count = 0

    def after_create_session(self, session, coord):  # pylint: disable=unused-argument
        self._tf_global_step = tf.train.get_global_step()
        assert self._tf_global_step, 'Internal error: RunMetadataHook cannot retrieve global step.'

    def before_run(self, run_context):  # pylint: disable=unused-argument
        self._should_trigger = self._timer.should_trigger_for_step(self._iter_count)
        if self._should_trigger:
            self._timer.update_last_triggered_step(self._iter_count)
            return tf.train.SessionRunArgs(
                fetches=self._tf_global_step,
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE))
        else:
            return None

    def after_run(self, run_context, run_values):  # pylint: disable=unused-argument
        if self._should_trigger:
            self._np_global_step = run_values.results
            # self._iter_count = self._np_global_step
            self._timer.update_last_triggered_step(self._iter_count)
            run_metadata = run_values.run_metadata
            if run_metadata is not None:
                tl = timeline.Timeline(run_metadata.step_stats)
                trace = tl.generate_chrome_trace_format()
                trace_filename = os.path.join(self._log_dir, f"tf_trace-{self._np_global_step}.json")
                tf.logging.info(f"Writing trace to {trace_filename}.")
                file_io.write_string_to_file(trace_filename, trace)
                # TODO: add run_metadata to summaries with summary_writer
                #   find how summaries are saved in the estimator and add them
                # summary_writer.add_run_metadata(run_metadata, f"run_metadata-{self._global_step}")

        self._iter_count += 1


def _have_compatible_shapes(lot):
    # lot: list_of_tensors
    tv = True
    for t1, t2 in itertools.combinations(lot, 2):
        tv = tv and t1.shape.is_compatible_with(t2.shape)
    return tv


def _have_equal_shapes(lot):
    # lot: list_of_tensors
    tv = True
    for t1, t2 in itertools.combinations(lot, 2):
        tv = tv and (t1.shape == t2.shape)
    return tv
