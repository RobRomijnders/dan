import copy, argparse, collections, json, os, glob, sys
from utils.utils import _replacevoids
from os.path import join


class SemanticSegmentationArguments(object):
    """Example class for how to collect arguments for command line execution.
    """
    _DEFAULT_PROBLEM = 'problem01'

    def __init__(self):
        self._parser = argparse.ArgumentParser()
        self.add_general_arguments()
        self.add_tf_arguments()

    def parse_args(self, argv):
        # parse all arguments and add manually additional arguments
        self.args = self._parser.parse_args(argv)

        # problem definitions
        self.args.training_problem_def = json.load(
            open(self.args.training_problem_def_path, 'r'))
        if hasattr(self.args, 'inference_problem_def_path'):
            if self.args.inference_problem_def_path is None:
                # TODO: implement the use of inference problem_def for training results in all functions
                self.args.inference_problem_def = self.args.training_problem_def
            else:
                self.args.inference_problem_def = json.load(
                    open(self.args.inference_problem_def_path, 'r'))
        if hasattr(self.args, 'evaluation_problem_def_path'):
            if self.args.evaluation_problem_def_path is None:
                self.args.evaluation_problem_def = self.args.training_problem_def
            else:
                self.args.evaluation_problem_def = json.load(
                    open(self.args.evaluation_problem_def_path, 'r'))
        if hasattr(self.args, 'additional_problem_def_path'):
            if self.args.additional_problem_def_path is None:
                self.args.additional_problem_def = self.args.training_problem_def
            else:
                self.args.additional_problem_def = json.load(
                    open(self.args.additional_problem_def_path, 'r'))


        # _validate_problem_config(self.args.config)
        # by convention class ids start at 0
        number_of_training_classes = max(self.args.training_problem_def['lids2cids']) + 1
        trained_with_void_class = -1 in self.args.training_problem_def['lids2cids']
        self.args.training_Nclasses = number_of_training_classes + trained_with_void_class

        self.args.training_lids2cids = _replacevoids(self.args.training_problem_def['lids2cids'])

        if hasattr(self.args, 'additional_problem_def'):
            self.args.additional_lids2cids = _replacevoids(self.args.additional_problem_def['lids2cids'])

        # define dict from objects' attributes so as they are ordered for printing alignment
        # self.args_dict = collections.OrderedDict(sorted(vars(self.args).items()))

        return self.args

    def add_general_arguments(self):
        # hs -> ||                                 SYSTEM                                       || -> hs/s
        # hs -> || hn -> ||                   LEARNABLE NETWORK                     || -> hn/sn || -> hs/s
        # hs -> || hn -> ||  hf -> FEATURE EXTRACTOR -> hf/sfe -> [UPSAMPLER -> hf] || -> hn/sn || -> hs/s
        # input || image || [tile] -> batch ->               supervision -> [stich] || labels    || output
        self._parser.add_argument('--stride_system', type=int, default=1,
                                  help='Output stride of the system. Use 1 for same input and output dimensions.')
        self._parser.add_argument('--stride_network', type=int, default=1,
                                  help='Output stride of the network. Use in case labels have different dimensions than output of learnable network.')
        self._parser.add_argument('--stride_feature_extractor', type=int, default=8,
                                  help='Output stride of the feature extractor. For the resnet_v1_* familly must be in {4,8,16,...}.')
        self._parser.add_argument('--name_feature_extractor', type=str, default='resnet_v1_50',
                                  choices=['resnet_v1_50', 'resnet_v1_101'], help='Feature extractor network.')
        self._parser.add_argument('--height_system', type=int, default=None,
                                  help='Height of input images to the system. If None arbitrary height is supported.')
        self._parser.add_argument('--width_system', type=int, default=None,
                                  help='Width of input images to the system. If None arbitrary width is supported.')
        self._parser.add_argument('--height_network', type=int, default=384,
                                  help='Height of input images to the trainable network.')
        self._parser.add_argument('--width_network', type=int, default=768,
                                  help='Width of input images to the trainable network.')
        self._parser.add_argument('--height_feature_extractor', type=int, default=384,
                                  help='Height of feature extractor images. If height_feature_extractor != height_network then it must be its divisor for patch-wise training.')
        self._parser.add_argument('--width_feature_extractor', type=int, default=768,
                                  help='Width of feature extractor images. If width_feature_extractor != width_network then it must be its divisor for patch-wise training.')

        # 1024 -> ||                                   ALGORITHM                                 || -> 1024  ::  h=512, s=512/512=1
        # 1024 -> || 1024 -> ||                     LEARNABLE NETWORK                 || -> 1024 || -> 1024  ::  hl=512, snet=512/512=1
        # 1024 -> || 1024 -> || 512 -> FEATURE EXTRACTOR -> 128 -> [UPSAMPLER -> 512] || -> 1024 || -> 1024  ::  hf=512, sfe=512/128=4
        self._parser.add_argument('--feature_dims_decreased', type=int, default=256,
                                  help='If >0 decreases feature dimensions of the feature extractor\'s output (usually 2048) to feature_dims_decreased using another convolutional layer.')
        self._parser.add_argument('--fov_expansion_kernel_size', type=int, default=0,
                                  help='If >0 increases the Field of View of the feature representation using an extra convolutional layer with this kernel.')
        self._parser.add_argument('--fov_expansion_kernel_rate', type=int, default=0,
                                  help='If >0 increases the Field of View of the feature representation using an extra convolutional layer with this dilation rate.')
        self._parser.add_argument('--upsampling_method', type=str, default='hybrid',
                                  choices=['no', 'bilinear', 'hybrid'],
                                  help='No, Bilinear or hybrid upsampling are currently supported.')

        self._parser.add_argument('--gradient_clip_norm', type=float, default=5.0,
                                  help='the clip norm for the clipping of gradients. Only used when a positive number')
        self._parser.add_argument('--regularize_extra', type=int, default=1,
                                  help='TEMPORARY HACKY IMPLEMENTATION to try different regularization schemes')
        # self._parser.add_argument('--subnet_experimenting', action='store_true', help='Temporary flag for subnet experimenting.')

    def add_tf_arguments(self):
        # general flags
        # self._parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'cityscapes_extended', 'cityscapes_gtsdb_0', 'gtsdb', 'cityscapes+gtsdb', 'cityscapes+cityscapes_extended'], help='Which dataset to use (3 first are tf records datasets the last is training with 2 datasets).')
        # self._parser.add_argument('--dataset_path', type=str, default='/media/panos/data/datasets/cityscapes', help='Cityscapes dataset path.')
        # self._parser.add_argument('--log_every_n_steps', type=int, default=100, help='The frequency, in terms of global steps, that the loss and global step and logged.')
        # self._parser.add_argument('--summarize_grads', default=False, action='store_true', help='Whether to mummarize gradients.')
        self._parser.add_argument('--enable_xla', action='store_true', help='Whether to enable XLA accelaration.')
        self._parser.add_argument('--custom_normalization_mode', type=str, default='batch',
                                  help='custom normalization strategy '
                                       'Batch Normalization (batch), '
                                       'Layer Normalization (layer), '
                                       'Instance Normalization (instance) or '
                                       'no normalization (none)')

    def add_train_arguments(self):
        """Arguments for training.

        TFRecords requirements...
        """
        # general configuration flags (in future will be saved in otherconfig)
        self._parser.add_argument('log_dir', type=str, default='...',
                                  help='Directory for saving checkpoints, settings, graphs and training statistics.')
        self._parser.add_argument('tfrecords_path', type=str,
                                  default='/media/panos/data/datasets/cityscapes/tfrecords/trainFine.tfrecords',
                                  help='Training is supported only from TFRecords. Refer to help for the mandatory fields for examples inside tfrecords.')
        self._parser.add_argument('--tfrecords_path_add', type=str,
                                  default=None, help='Additional tfrecord for training with two datasets')
        self._parser.add_argument('Ntrain', type=int, default=2975,
                                  help='Temporary parameter for the number of training examples.')
        self._parser.add_argument('--init_ckpt_path', type=str,
                                  default='',
                                  help='If provided and log_dir is empty, same variables between checkpoint and the model will be initiallized from this checkpoint. Otherwise, training will continue from the latest checkpoint in log_dir according to tf.Estimator. If you want to initialize partially from this checkpoint delete of modify names of variables in the checkpoint.')
        self._parser.add_argument('training_problem_def_path', type=str,
                                  default='/media/panos/data/datasets/cityscapes/scripts/panos/jsons/' + SemanticSegmentationArguments._DEFAULT_PROBLEM + '.json',
                                  help='Problem definition json file. For required fields refer to help.')
        self._parser.add_argument('--inference_problem_def_path', type=str, default=None,
                                  help='Problem definition json file for inference. If provided it will be used instead of training_problem_def for inference. For required fields refer to help.')
        self._parser.add_argument('--save_checkpoints_steps', type=int, default=2000,
                                  help='Save checkpoint every save_checkpoints_steps steps.')
        self._parser.add_argument('--save_summaries_steps', type=int, default=120,
                                  help='Save summaries every save_summaries_steps steps.')
        # self._parser.add_argument('--collage_image_summaries', action='store_true', help='Whether to collage input and result images in summaries. Inputs and results won\'t be collaged if they have different sizes.')

        # optimization and losses flags (in future will be saved in hparams)
        self._parser.add_argument('--Ne', type=int, default=17, help='Number of epochs to train for.')
        self._parser.add_argument('--Nb', type=int, default=1, help='Number of examples per batch.')
        self._parser.add_argument('--learning_rate_schedule', type=str, default='schedule1',
                                  choices=['schedule1', 'schedule2'], help='Learning rate schedule.')
        self._parser.add_argument('--learning_rate_initial', type=float, default=0.01, help='Initial learning rate.')
        self._parser.add_argument('--learning_rate_decay', type=float, default=0.5,
                                  help='Decay rate for learning rate.')
        self._parser.add_argument('--optimizer', type=str, default='SGDM', choices=['SGD', 'SGDM'],
                                  help='Stochastic Gradient Descent optimizer with or without Momentum.')
        self._parser.add_argument('--batch_norm_decay', type=float, default=0.96,
                                  help='BatchNorm decay (decrease when using smaller batch (Nb or image dims)).')
        self._parser.add_argument('--batch_norm_istraining', type=bool, default=True,
                                  help='Indicator to use batch normalization in the feature extractor')
        self._parser.add_argument('--ema_decay', type=float, default=-1.0,
                                  help='If >0 additionally save exponential moving averages of training variables with this decay rate.')
        self._parser.add_argument('--regularization_weight', type=float, default=0.00017,
                                  help='Weight for the L2 regularization losses in the total loss (decrease when using smaller batch (Nb or image dims)).')
        # only for Momentum optimizer
        self._parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGDM.')
        self._parser.add_argument('--use_nesterov', action='store_true', help='Enable Nesterov acceleration for SGDM.')

        # For the adversarial training
        self._parser.add_argument('--switch_train_op', type=bool, default=False,
                                  help='boolean variable to indicate if you want to switch between training ops. Only works when two or more datasets are fed')
        self._parser.add_argument('--unsupervised_adaptation', type=bool, default=False,
                                  help='boolean variable to indicate if you want unsupervised adaptation. Only works when two or more datasets are fed')
        self._parser.add_argument('--lambda_conf', type=float, default=0.1,
                                  help='weighing coefficients for the confusion loss')
        self._parser.add_argument('--switch_period', type=int, default=1,
                                  help='period for which to switch the alternation of training ops')
        self._parser.add_argument('--dom_class_type', type=int, default=1,
                                  help='period for which to switch the alternation of training ops')
        self._parser.add_argument('--confusion_loss', type=str, default='target',
                                  help='Defines the confusion loss, can be "target" or "cross"')
        self._parser.add_argument('--ramp_start', type=int, default=3000,
                                  help='step at which to start ramping up the lambda conf')
        self._parser.add_argument('--additional_problem_def_path', type=str,
                                  help='Problem definition for the additional dataset. If empty, the training_problem_def will be used', default=None)


    def add_predict_arguments(self):

        # saved model arguments: log_dir, [ckppt_path], training_problem_def_path
        self._parser.add_argument('log_dir', type=str, default=None,
                                  help='Logging directory containing the trained model checkpoints and settings. The latest checkpoint will be loaded from this directory by default, unless ckpt_path is provided.')
        self._parser.add_argument('--ckpt_path', type=str, default=None,
                                  help='If provided, this checkpoint (if exists) will be used.')
        self._parser.add_argument('training_problem_def_path', type=str,
                                  default='/media/panos/data/datasets/cityscapes/scripts/panos/jsons/' + SemanticSegmentationArguments._DEFAULT_PROBLEM + '.json',
                                  help='Problem definition json file that the model is trained with. For required fields refer to help.')

        # inference arguments: prediction_dir, [results_dir], [inference_problem_def_path],
        #                      [plotting], [export_color_images], [export_lids_images]
        self._parser.add_argument('predict_dir', type=str, default=None,
                                  help='Directory to scan for media recursively and to write results under created results directory with the same directory structure. For supported media files check help.')
        self._parser.add_argument('--results_dir', type=str, default=None,
                                  help='If provided results will be written to this directory.')
        self._parser.add_argument('--inference_problem_def_path', type=str, default=None,
                                  help='Problem definition json file for inference. If provided it will be used instead of training_problem_def. For required fields refer to help.')
        self._parser.add_argument('--plotting', action='store_true',
                                  help='Whether to plot results.')
        self._parser.add_argument('--timeout', type=float, default=10.0,
                                  help='Timeout for continuous plotting, if plotting flag is provided.')
        self._parser.add_argument('--export_color_images', action='store_true',
                                  help='Whether to export color image results.')
        self._parser.add_argument('--export_lids_images', action='store_true',
                                  help='Whether to export label ids image results. Label ids are defined in {training,inference}_problem_def_path.')
        self._parser.add_argument('--replace_void_decisions', action='store_true',
                                  help='Whether to replace void labeled pixels with the second most probable class (effective only when void (-1) is provided in lids2cids field in training problem definition). Enable only for official prediction/evaluation as it uses a time consuming function.')

        # SemanticSegmentation and system arguments:  [Nb], [restore_emas]
        self._parser.add_argument('--Nb', type=int, default=1,
                                  help='Number of examples per batch.')
        self._parser.add_argument('--restore_emas', action='store_true',
                                  help='Whether to restore exponential moving averages instead of normal last step\'s saved variables.')

        # consider for adding in the future arguments
        # self._parser.add_argument('--export_probs', action='store_true', help='Whether to export probabilities results.')
        # self._parser.add_argument('--export_for_algorithm_evaluation', action='store_true', help='Whether to plot and export using the algorithm input size (h,w).')

    def add_evaluate_arguments(self):
        self._parser.add_argument('log_dir', type=str, default=None,
                                  help='Logging directory containing the trained model checkpoints and settings. The latest checkpoint will be evaluated from this directory by default, unless ckpt_path or evall_all_ckpts are provided.')
        self._parser.add_argument('--eval_all_ckpts', type=int, default=1,
                                  help='Whether to evaluate more checkpoints in log_dir. It has priority over --ckpt_path argument. The number specifies number of checkpoints')
        self._parser.add_argument('--ckpt_path', type=str, default=None,
                                  help='If provided, this checkpoint (if exists) will be evaluated.')
        self._parser.add_argument('tfrecords_path', type=str,
                                  default='/media/panos/data/datasets/cityscapes/tfrecords/valFine.tfrecords',
                                  help='Evaluation is supported only from TFRecords. Refer to help for the mandatory fields for examples inside tfrecords.')
        self._parser.add_argument('Neval', type=int, default=500,
                                  help='Temporary parameter for the number of evaluated examples.')
        self._parser.add_argument('training_problem_def_path', type=str,
                                  default='/home/panos/tensorflow/panos/panos/ResNet_v1/jsons/' + self._DEFAULT_PROBLEM + '.json',
                                  help='Problem definition json file that the model is trained with. For required fields refer to help.')
        self._parser.add_argument('--evaluation_problem_def_path', type=str, default=None,
                                  help='Problem definition json file for evaluation. If provided it will be used instead of training_problem_def. For required fields refer to help.')

        # self._parser.add_argument('eval_steps', type=int, help='500 for cityscapes val, 2975 for cityscapes train.')
        self._parser.add_argument('--Nb', type=int, default=1,
                                  help='Number of examples per batch.')
        self._parser.add_argument('--restore_emas', action='store_true',
                                  help='Whether to restore exponential moving averages instead of normal last step\'s saved variables.')
        self._parser.add_argument('--confusion_matrix_filename', type=str, default=None)

        # consider for future
        # self._parser.add_argument('--results_dir', type=str, default=None,
        #                           help='If provided evaluation results will be written to this directory, otherwise they will be written under a created results directory under log_dir.')

    def add_extract_arguments(self):

        # saved model arguments: log_dir, [ckppt_path], training_problem_def_path
        self._parser.add_argument('log_dir', type=str, default=None,
                                  help='Logging directory containing the trained model checkpoints and settings. The latest checkpoint will be loaded from this directory by default, unless ckpt_path is provided.')
        self._parser.add_argument('--ckpt_path', type=str, default=None,
                                  help='If provided, this checkpoint (if exists) will be used.')
        self._parser.add_argument('training_problem_def_path', type=str,
                                  default='/media/panos/data/datasets/cityscapes/scripts/panos/jsons/' + SemanticSegmentationArguments._DEFAULT_PROBLEM + '.json',
                                  help='Problem definition json file that the model is trained with. For required fields refer to help.')

        # inference arguments: prediction_dir, [results_dir], [inference_problem_def_path],
        #                      [plotting], [export_color_images], [export_lids_images]
        self._parser.add_argument('predict_dir', type=str, default=None,
                                  help='Directory to scan for media recursively and to write results under created results directory with the same directory structure. For supported media files check help.')
        self._parser.add_argument('--results_dir', type=str, default=None,
                                  help='If provided results will be written to this directory.')
        self._parser.add_argument('--inference_problem_def_path', type=str, default=None,
                                  help='Problem definition json file for inference. If provided it will be used instead of training_problem_def. For required fields refer to help.')
        self._parser.add_argument('--plotting', action='store_true',
                                  help='Whether to plot results.')
        self._parser.add_argument('--timeout', type=float, default=10.0,
                                  help='Timeout for continuous plotting, if plotting flag is provided.')
        self._parser.add_argument('--export_color_images', action='store_true',
                                  help='Whether to export color image results.')
        self._parser.add_argument('--export_lids_images', action='store_true',
                                  help='Whether to export label ids image results. Label ids are defined in {training,inference}_problem_def_path.')
        self._parser.add_argument('--replace_void_decisions', action='store_true',
                                  help='Whether to replace void labeled pixels with the second most probable class (effective only when void (-1) is provided in lids2cids field in training problem definition). Enable only for official prediction/evaluation as it uses a time consuming function.')

        # SemanticSegmentation and system arguments:  [Nb], [restore_emas]
        self._parser.add_argument('--Nb', type=int, default=1,
                                  help='Number of examples per batch.')
        self._parser.add_argument('--restore_emas', action='store_true',
                                  help='Whether to restore exponential moving averages instead of normal last step\'s saved variables.')

        # More arguments
        self._parser.add_argument('--num_images', type=int, default=100,
                                  help='Number of images to extract the representations from')
        self._parser.add_argument('--eval_all_ckpts', type=int, default=1,
                                  help='Whether to evaluate more checkpoints in log_dir. It has priority over --ckpt_path argument. The number specifies number of checkpoints')
        self._parser.add_argument('--num_assignments', type=int, default=200,
                                  help='Number of new assignments to make when adapting the batch norm statistics')


def set_bn_cancellation(args):
    args.batch_norm_cancellation = False
    with open(join(args.log_dir, 'settings.txt')) as f:
        for line in f:
            if 'batch_norm_cancellation' in line:
                if 'True' in line:
                    args.batch_norm_cancellation = True
                    args.custom_normalization_mode = 'none'
                return args
            if 'custom_normalization_mode' in line:
                args.custom_normalization_mode = line.split(':')[2].strip()
                return args
    args.custom_normalization_mode = 'batch'
    return args
