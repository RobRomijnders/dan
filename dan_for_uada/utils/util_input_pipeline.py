"""Input pipelines utilities, common code."""

import tensorflow as tf


# from tensorflow.python.eager import context

def adjust_images_labels_size(images, labels, size):
    """
    Smartly adjust `images` and `labels` spatial dimensions to `size`.
    The input of an FCN feature extractor for semantic segmentation must normally be of
      fixed spatial dimensions. This function adjusts the spatial dimensions of input images
      and corresponding labels to a fixed predefined size. The "smart adjustment" is done
      according to the following rules. First the input is resized, preserving the aspect
      ratio, to the smallest size, which is at least bigger than `size` and then cropped to `size`.
    There are 9 possibilities for comparing the dimensions of dataset images (H, W)
    and feature extractor (h, w). They stem from permutations with repetition (n_\Pi_r = n^r)
    of n = 3 objects (<, >, =) taken by r = 2 (h ? H, w ? W).
        H ? h   W ? w                     H ? h   W ? w
          =       =    random crop          =       <    upscale and random crop
          =       >    random crop          <       =    upscale and random crop
          >       =    random crop          >       <    upscale and random crop
                                            <       <    upscale and random crop
                                            <       >    upscale and random crop
                                            >       >    downscale and random crop
    These cases can be summarized to the following pseudocode,
      which creates the smallest possible D' >= d, where d = (h, w), D = (H, W):
        if reduce_any(D < d) or reduce_all(D > d):
          upscale to D' = D * reduce_max(d/D)
        random crop
    These transformations preserve the relative scale of objects across different image sizes and
      are "meaningful" for training a network in combination with adjusted sized inference.

    Arguments:
      images: tf.float32, (?, ?, ?, 3), in [0, 1)
      labels: tf.int32, (?, ?, ?)
      size: Python tuple, (2,), a valid spatial size in height, width order

    Return:
      images, labels: ...
    """

    assert isinstance(size, tuple), 'size must be a tuple.'
    # TODO(panos): add input checks

    shape = tf.shape(images)
    spatial_shape = shape[1:3]

    upscale_condition = tf.reduce_any(tf.less(spatial_shape, size))
    downscale_condition = tf.reduce_all(tf.greater(spatial_shape, size))
    factor = tf.cast(tf.reduce_max(size / spatial_shape), tf.float32)

    def _resize_images_labels(images, labels, size):
        images = tf.image.resize_images(images, size)
        labels = tf.image.resize_images(labels[..., tf.newaxis],
                                        size,
                                        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)[..., 0]
        return images, labels

    def _true_fn():
        return _resize_images_labels(
            images,
            labels,
            tf.cast(tf.ceil(factor * tf.cast(spatial_shape, tf.float32)), tf.int32))

    def _false_fn():
        return images, labels

    combined_condition = tf.logical_or(upscale_condition, downscale_condition)
    images, labels = tf.cond(combined_condition, _true_fn, _false_fn)

    def _random_crop(images, labels, size):
        # TODO(panos): check images and labels spatial size statically,
        #   if it is defined and is the same as size do not add random_crop ops
        # convert images to tf.int32 to concat and random crop the same area
        images = tf.cast(tf.image.convert_image_dtype(images, tf.uint8), tf.int32)
        concated = tf.concat([images, labels[..., tf.newaxis]], axis=3)
        Nb = images.shape[0].value
        crop_size = (Nb,) + size + (4,)
        print('debug:concated,crop_size:', concated, crop_size)
        concated_cropped = tf.random_crop(concated, crop_size)
        # convert images back to tf.float32
        images = tf.image.convert_image_dtype(tf.cast(concated_cropped[..., :3], tf.uint8), tf.float32)
        labels = concated_cropped[..., 3]
        return images, labels

    images, labels = _random_crop(images, labels, size)

    return images, labels


def from_0_1_to_m1_1(images):
    """
    Center images from [0, 1) to [-1, 1).

    Arguments:
      images: tf.float32, in [0, 1), of any dimensions

    Return:
      images linearly scaled to [-1, 1)
    """

    # TODO(panos): generalize to any range
    # shifting from [0, 1) to [-1, 1) is equivalent to assuming 0.5 mean
    mean = 0.5
    proimages = (images - mean) / mean

    return proimages


# TODO(panos): as noted in MirrorStrategy, in a multi-gpu setting the effective
#   batch size is num_gpus * Nb, so deal with this until per batch broadcasting
#   is implemented in core tensorflow package
# by default all available gpus of the machine are used
def get_temp_Nb(runconfig, Nb):
    div, mod = divmod(Nb, runconfig.distribution.num_towers())
    assert not mod, 'for now Nb must be divisible by the number of available GPUs.'
    return div