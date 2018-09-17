"""
Semantic Segmentation system.
"""

import copy
import functools
import glob
import collections
from os.path import join, split, exists, isdir
from os import makedirs
import numpy as np
from PIL import Image
from estimator.define_estimator import define_estimator
from utils.utils import _replacevoids, print_metrics_from_confusion_matrix, maybe_makedirs
from utils.utils import get_all_ckpt_paths
import tensorflow as tf
import logging

tf.logging.set_verbosity(tf.logging.DEBUG)

log = logging.getLogger('semantic_segmentation')


class SemanticSegmentation(object):
    """A Semantic Segmentation system class.

    This class uses the tf.estimator API for reproducibility and transferability
    of experiments, as well as to take advantage of automatic parallelization
    (across CPUs, GPUs) that this API provides.

    Components of this class:
      * model for per-pixel semantic segmentation: provided by model_fn
      * input functions for different modes: provided by input_fns
      * settings for the whole system: provided by settings

    Provides the following functions to the user:
      * train
      * evaluate
      * predict

    More info to be added soon...

    """

    def __init__(self, input_fns, model_fn, settings=None):
        """Constructs a SS instance... (More info to be added...)

        Args:
          input_fns: a dictionary containing 'train', 'eval' and 'predict' keys to corresponding
            input functions. The input functions will be called by the respective functions of this
            class with the following signature: (customconfig, params) and should
            return a tuple of (features, labels) containing feature and label dictionaries with
            the needed str-tf.Tensor pairs. (required keys will be added, for now check code...).
          model_fn: model function for the fully convolutional semantic segmentation model chosen.
            It will be called with the following signature: (mode, features, labels, config, params).
          settings: an object containing all parsed parameters from command line as attributes.
          [UNUSED ARGUMENTS FOR NOW]
        Comments / design choices:
          1) Everytime a method {train, predict, evaluate} of this object is called a new local
             estimator is created with the desired properties saved in this class' members, specified
             to the respective action. This choice is made purely for memory efficiency.
        """
        assert settings is not None, (
            'settings must be provided for now.')
        _validate_settings(settings)

        self._input_fns = input_fns
        self._model_fn = model_fn
        self._settings = copy.deepcopy(settings)
        self._estimator = None

        # by convention class ids start at 0
        number_of_training_classes = max(self._settings.training_problem_def['lids2cids']) + 1
        trained_with_void_class = -1 in self._settings.training_problem_def['lids2cids']  # -1 indicates a void class
        self._settings.training_Nclasses = number_of_training_classes + trained_with_void_class

        self._settings.training_lids2cids = _replacevoids(
            self._settings.training_problem_def['lids2cids'])

        # save to settings for external access
        self._settings.eval_res_dir = make_evaluation_dir(settings.log_dir)

    @property
    def settings(self):
        return self._settings

    @property
    def eval_res_dir(self):
        return self._settings.eval_res_dir

    def _create_estimator(self, runconfig):
        self._estimator = tf.estimator.Estimator(
            functools.partial(define_estimator, model_fn=self._model_fn),
            model_dir=self._settings.log_dir,
            config=runconfig,
            params=self._settings)
        return self._estimator

    def get_all_model_checkpoint_paths(self):
        """
        Finds all the checkpoint paths from the log_dir

        Note that one could also use tf.train.get_checkpoint_state(). However, that will
        not work when the network is retrained or adapted. Therefore, we parse the filenames manually
        :return:
        """
        if self._settings.eval_all_ckpts > 1:
            # First find all checkpoint paths
            all_ckpt_paths = get_all_ckpt_paths(self._settings.log_dir)
            all_model_checkpoint_paths = list(all_ckpt_paths)[-self._settings.eval_all_ckpts:]
        else:
            all_model_checkpoint_paths = [self._settings.ckpt_path]
        log.debug(f"\n{len(all_model_checkpoint_paths)} checkpoint(s) will be evaluated.\n")
        return all_model_checkpoint_paths

    def train(self):
        """Train the Semantic Segmentation model.
        """

        # create log dir
        if not tf.gfile.Exists(self._settings.log_dir):
            tf.gfile.MakeDirs(self._settings.log_dir)
            log.debug('Created new logging directory:', self._settings.log_dir)

        write_settings_to_file(self._settings)

        # define the session_config
        if self._settings.enable_xla:
            session_config = tf.ConfigProto()
            # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
            # session_config = tf.ConfigProto(log_device_placement=True, gpu_options=gpu_options)
            # session_config.log_device_placement = True
            # session_config.gpu_options.per_process_gpu_memory_fraction = 0.95
            session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        else:
            session_config = None

        # Tensorflow internal error till at least r1.4
        # if keep_checkpoint_max is set to 0 or None doesn't do what it is supposed to do from docs
        runconfig = tf.estimator.RunConfig(
            model_dir=self._settings.log_dir,
            save_summary_steps=self._settings.save_summaries_steps,
            save_checkpoints_steps=self._settings.save_checkpoints_steps,
            session_config=session_config,
            keep_checkpoint_max=None,
            log_step_count_steps=self._settings.save_summaries_steps)

        # create a local estimator
        self._create_estimator(runconfig)

        return self._estimator.train(
            input_fn=self._input_fns['train'],
            max_steps=self._settings.num_training_steps)

    def get_predictions(self):
        """
        Gets the predictions from the estimator
        :return:
        """
        if self._settings.Nb > 1:
            log.debug('\nWARNING: during prediction only images with same shape (size and channels) '
                      'are supported for batch size greater than one. In case of runtime error '
                      'change batch size to 1.\n')
        if self._settings.enable_xla:
            session_config = tf.ConfigProto()
            session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        else:
            session_config = None

        runconfig = tf.estimator.RunConfig(
            model_dir=self._settings.log_dir,
            session_config=session_config)

        self._create_estimator(runconfig)

        predict_keys = copy.deepcopy(self._settings.predict_keys)
        # if void exists probabilities are needed (see _replace_void_labels)
        void_exists_lids2cids = -1 in self._settings.training_problem_def['lids2cids']
        if void_exists_lids2cids:
            predict_keys.append('probabilities')

        predictions = self._estimator.predict(
            input_fn=self._input_fns['predict'],
            predict_keys=predict_keys,
            # if None latest checkpoint in self._settings.model_dir will be used
            checkpoint_path=self._settings.ckpt_path)
        return predictions

    # def predict(self):
    #     predictions = self.get_predictions()
    #
    #     # TODO: resize to system dimensions for the outputs
    #     void_exists_lids2cids = -1 in self._settings.training_problem_def['lids2cids']
    #     # deal with void in training lids2cids
    #     if void_exists_lids2cids:
    #         if self._settings.replace_void_decisions:
    #             predictions = self._replace_void_labels(predictions)
    #         else:
    #             def _print_warning(predictions):
    #                 for prediction in predictions:
    #                     void_exists_decs = np.any(np.equal(prediction['decisions'],
    #                                                        max(self._settings.training_lids2cids)))
    #                     if void_exists_decs:
    #                         log.debug(f"\nWARNING: void class label ({max(self._settings.training_lids2cids)}) "
    #                                   "exists in decisions.\n")
    #                     yield prediction
    #
    #             predictions = _print_warning(predictions)
    #
    #         # delete 'probabilities' key since it was needed only locally for sanity
    #         def _delete_probs(predictions):
    #             for prediction in predictions:
    #                 del prediction['probabilities']
    #                 yield prediction
    #
    #         predictions = _delete_probs(predictions)
    #
    #     # deal with void in inference cids2lids
    #     if -1 in self._settings.inference_problem_def['cids2lids']:
    #         log.debug('\nWARNING: -1 exists in cids2lids field of inference problem definition. '
    #                   'For now it must me handled externally, and may cause outputs to have '
    #                   '-1 labels.\n')
    #
    #     # if predicting for different problem definition additional mapping is needed
    #     if self._settings.training_problem_def != self._settings.inference_problem_def:
    #         predictions = self._map_predictions_to_inference_problem_def(predictions)
    #
    #     # resize to system dimensions: the output should have the provided system spatial size
    #     predictions = self._resize_decisions(predictions)
    #
    #     return predictions

    def evaluate(self):
        log.debug(f"\nWriting results in {self._settings.eval_res_dir}.\n")
        maybe_makedirs(self._settings.eval_res_dir)

        write_settings_to_file(self._settings, self._settings.eval_res_dir)

        if self._settings.enable_xla:
            session_config = tf.ConfigProto()
            session_config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        else:
            session_config = None

        runconfig = tf.estimator.RunConfig(
            model_dir=self._settings.log_dir,
            session_config=session_config,
            keep_checkpoint_max=2)

        # create a local estimator
        self._create_estimator(runconfig)

        # get labels needed for online printing
        labels = self._settings.evaluation_problem_def['cids2labels']
        void_exists = -1 in self._settings.evaluation_problem_def['lids2cids']
        labels = labels[:-1] if void_exists else labels

        def yield_all_metrics():
            for i, checkpoint_path in enumerate(self.get_all_model_checkpoint_paths()):
                log.debug('Checkpoint %i/%i' % (i, self._settings.eval_all_ckpts))
                # metrics contains only confusion matrix for now (and loss and global step)
                metrics = self._estimator.evaluate(
                    input_fn=self._input_fns['eval'],
                    steps=self._settings.num_eval_steps,
                    # if None latest in model_dir will be used
                    checkpoint_path=checkpoint_path,
                    name=split(self._settings.eval_res_dir)[1][-2:])

                # deal with void in evaluation lids2cids
                if -1 in self._settings.evaluation_problem_def['lids2cids']:
                    assert set(metrics.keys()) == {'global_step', 'loss', 'confusion_matrix'}, (
                        'internal error: only confusion matrix metric is supported for mapping to '
                        'a new problem definition for now. Change to training problem definition.')
                    metrics['confusion_matrix'] = metrics['confusion_matrix'][:-1, :-1]

                # transform to different evaluation problem definition
                # if self._settings.training_problem_def != self._settings.evaluation_problem_def:
                #     metrics = self._map_metrics_to_evaluation_problem_def(metrics)
                #
                #     # deal with void in training_cids2evaluation_cids
                #     if -1 in self._settings.evaluation_problem_def['training_cids2evaluation_cids']:
                #         assert set(metrics.keys()) == {'global_step', 'loss', 'confusion_matrix'}, (
                #             'internal error: only confusion matrix metric is supported for mapping to '
                #             'a new problem definition for now. Change to training problem definition.')
                #         metrics['confusion_matrix'] = metrics['confusion_matrix'][:-1, :-1]

                # online print the summary of metrics to terminal
                print_metrics_from_confusion_matrix(metrics['confusion_matrix'], labels)
                yield metrics
        return list(yield_all_metrics())

    def _replace_void_labels(self, predictions):
        # WARNING: time consuming function (due to argpartition), can take from 300 to 800 ms
        # enable it only when predicting for official evaluation/prediction

        # if void (-1) is provided in lids2cids, then the pixels that are predicted to belong
        # to the (internally) added void class should be labeled with the second most probable class
        # only decisions field is suppported for now
        accepted_keys = {'probabilities', 'decisions', 'rawimages', 'rawimagespaths'}
        predict_keys = self._settings.predict_keys
        # since 'probabilities' key is added temporarily it doesn't exist in _settings.predict_keys
        predict_keys.append('probabilities')
        for prediction in predictions:
            assert set(prediction.keys()).intersection(
                set(predict_keys)) == accepted_keys, (
                'internal error: only \'decisions\' predict_key is supported for mapping to '
                'a new problem definition for now. Change to training problem definition.')
            old_decs = prediction['decisions']  # 2D
            old_probs = prediction['probabilities']  # 3D
            void_decs_mask = np.equal(old_decs, max(self._settings.training_lids2cids))
            # implementing: values, indices = tf.nn.top_k(old_probs, k=2) # 3D
            # in numpy using argpartition for indices and
            # mesh grid and advanced indexing for values
            # argpartition returns np.int64
            top2_indices = np.argpartition(old_probs, -2)[..., -2]  # 3D -> 2D
            # row, col = np.mgrid[0:old_probs.shape[0], 0:old_probs.shape[1]] # 2D, 2D
            # values = old_probs[row, col, indices]
            new_decs = np.where(void_decs_mask,
                                top2_indices.astype(np.int32, casting='same_kind'),
                                old_decs)
            prediction['decisions'] = new_decs
            yield prediction

    def _map_predictions_to_inference_problem_def(self, predictions):
        assert 'training_cids2inference_cids' in self._settings.inference_problem_def.keys(), (
            'Inference problem definition should have training_cids2inference_cids field, '
            'since provided inference problem definition file is not the same as training '
            'problem definition file.')

        tcids2pcids = np.array(_replacevoids(
            self._settings.inference_problem_def['training_cids2inference_cids']))

        for prediction in predictions:
            # only decisions is suppported for now
            assert set(prediction.keys()).intersection(
                set(self._settings.predict_keys)) == {'decisions', 'rawimages', 'rawimagespaths'}, (
                'internal error: only decisions predict_key is supported for mapping to '
                'a new problem definition for now. Change to training problem definition.')

            old_decisions = prediction['decisions']

            # TODO: add type and shape assertions
            assert old_decisions.ndim == 2, f"internal error: decisions shape is {old_decisions.shape}."

            new_decisions = tcids2pcids[old_decisions]
            if np.any(np.equal(new_decisions, -1)):
                log.debug('WARNING: -1 label exists in decisions, handle it properly externally.')
                # raise NotImplementedError(
                #     'void mapping in different inference problem def is not yet implemented.')

            prediction['decisions'] = new_decisions

            yield prediction

    def _map_metrics_to_evaluation_problem_def(self, metrics):
        # if a net should be evaluated with problem that is not the problem with which it was
        # trained for, then the mappings from that problem should be provided.

        # only confusion_matrix is suppported for now
        assert set(metrics.keys()) == {'global_step', 'loss', 'confusion_matrix'}, (
            'internal error: only confusion matrix metric is supported for mapping to'
            'a new problem definition for now. Change to training problem definition.')
        assert 'training_cids2evaluation_cids' in self._settings.evaluation_problem_def.keys(), (
            'Evaluation problem definition should have training_cids2evaluation_cids field.')

        old_cm = metrics['confusion_matrix']
        tcids2ecids = np.array(_replacevoids(
            self._settings.evaluation_problem_def['training_cids2evaluation_cids']))

        # TODO: confusion matrix type and shape assertions
        assert old_cm.shape[0] == tcids2ecids.shape[
            0], 'Mapping lengths should me equal. %i (old_cm) %i (tcids2ecids)' % \
                (old_cm.shape[0], tcids2ecids.shape[0])

        temp_shape = (max(tcids2ecids) + 1, old_cm.shape[1])
        temp_cm = np.zeros(temp_shape, dtype=np.int64)

        # mas noiazei to kathe kainourio apo poio palio pairnei:
        #   i row of the new cm takes from rows of the old cm with indices:from_indices
        for i in range(temp_shape[0]):
            from_indices = [k for k, x in enumerate(tcids2ecids) if x == i]
            # print(from_indices)
            for fi in from_indices:
                temp_cm[i, :] += old_cm[fi, :].astype(np.int64)

        # oi grammes athroistikan kai tora tha athroistoun kai oi stiles
        new_shape = (max(tcids2ecids) + 1, max(tcids2ecids) + 1)
        new_cm = np.zeros(new_shape, dtype=np.int64)
        for j in range(new_shape[1]):
            from_indices = [k for k, x in enumerate(tcids2ecids) if x == j]
            # print(from_indices)
            for fi in from_indices:
                new_cm[:, j] += temp_cm[:, fi]

        metrics['confusion_matrix'] = new_cm

        return metrics

    def _resize_decisions(self, predictions):
        # resize decisions to system or input dimensions
        # only decisions is suppported for now
        # TODO: find a fast method without using PIL upsampling

        # new size defaults to provided values
        # if at least one is None then new size is the arbitrary size of rawimage in in step
        new_size = (self._settings.height_system, self._settings.width_system)
        is_arbitrary = not all(new_size)

        for prediction in predictions:
            assert set(prediction.keys()).intersection(
                set(self._settings.predict_keys)) == {'decisions', 'rawimages', 'rawimagespaths'}, (
                'internal error: only decisions predict_key is supported for mapping to '
                'a new problem definition for now. Change to training problem definition.')

            old_decisions = prediction['decisions']

            # TODO: add type and shape assertions
            assert old_decisions.ndim == 2, f"internal error: decisions shape is {old_decisions.shape}."

            if is_arbitrary:
                new_size = prediction['rawimages'].shape[:2]

            # save computation by comparing size
            if new_size != old_decisions.shape[:2]:
                new_decs = Image.fromarray(old_decisions).resize(reversed(new_size),
                                                                 resample=Image.NEAREST)
                prediction['decisions'] = np.array(new_decs)

            yield prediction


def _validate_settings(settings):
    # TODO: add more validations

    assert settings.stride_system == 1 and settings.stride_network == 1, (
        'For now only stride of 1 is supported for stride_{system, network}.')

    assert all([settings.height_network == settings.height_feature_extractor,
                settings.width_network == settings.width_feature_extractor]), (
        'For now {height, width}_{network, feature_extractor} should be equal.')

    # prediction specific
    if hasattr(settings, 'export_lids_images') and hasattr(settings, 'export_color_images'):
        if settings.export_lids_images or settings.export_color_images:
            assert settings.results_dir is not None and isdir(settings.results_dir), (
                'results_dir must a valid path if export_{lids, color}_images flags are True.')


def make_evaluation_dir(log_dir):
    """
    construct candidate path for evaluation results directory in log directory,
    with a unique counter index, e.g. if in log_dir/eval dir there exist
    eval00, eval01, eval02, eval04 dirs it will create a new dir named eval05
    TODO: better handle and warn for assumptions
    for now it assums that only eval_ with 2 digits are present
    :param log_dir:
    :return:
    """
    existing_eval_dirs = list(filter(isdir, glob.glob(join(log_dir, 'eval_*'))))
    if existing_eval_dirs:
        existing_eval_dirs_names = [split(ed)[1] for ed in existing_eval_dirs]
        max_cnt = max([int(edn[-2:]) for edn in existing_eval_dirs_names])
    else:
        max_cnt = -1
    return join(log_dir, 'eval_' + f"{max_cnt + 1:02}")


def write_settings_to_file(settings, log_dir_provided=None):
    """
    Writes all the settings to a file 'settings.txt' in the log_dir
    :param settings:
    :param log_dir_provided
    :return:
    """

    log_dir = log_dir_provided if log_dir_provided else settings.log_dir
    # vars(args).items() returns (key,value) tuples from args.__dict__
    # and sorted uses first element of tuples to sort
    settings_dict = collections.OrderedDict(sorted(vars(settings).items()))

    maybe_makedirs(log_dir, force_dir=True)

    # write configuration for future reference
    settings_filename = join(log_dir, 'settings.txt')
    # assert not exists(settings_filename), (f"Previous settings.txt found in "
    #                                        f"{log_dir}. Rename it manually and restart training.")
    with open(settings_filename, 'w') as f:
        for k, v in enumerate(settings_dict):
            print(f"{k:2} : {v} : {settings_dict[v]}", file=f)
