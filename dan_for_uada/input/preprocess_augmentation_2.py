
import numpy as np
from scipy import misc
import functools
import cv2
from tensorflow.python.ops import control_flow_ops
import tensorflow as tf

def preprocess_train(images, labels, params=None):
  # this function is called only once, while generating the input pipeline, so casing must use TF
  # images: TF tensor: B x hf x wf x 3, tf.float32 in [0,1)
  # labels: TF tensor: B x hf x wf (in case of upsampling), tf.int32, [0,Nclasses] (in case of extra void class)
  # proimages: TF tensor: B x hf x wf x 3, tf.float32, in [-1,1)
  # prolabels: TF tensor: B x hf x wf (in case of upsampling), tf.int32, [0,Nclasses] (in case of extra void class)
  
  if all([params,
          hasattr(params, 'tmp_from_dataset'),
          params.tmp_from_dataset=='cityscapes']):
    images, prolabels = random_color(images, labels, params)
    images, labels = random_blur(images, labels, params)
  else:
    print('\n WARNING: no dataset augmentation is done. \n')
  
  mean=.5
  proimages = (images - mean)/mean
  prolabels = labels
  
  return proimages, prolabels

def inverse_preprocess_train(proimages, prolabels, params=None):
  mean=.5
  images = (proimages*mean)+mean
  labels = prolabels

  return images, labels

def preprocess_predict(frame, params):
  # frame: numpy ndarray: h x w x ..., np.uint8
  # TODO: what happens on grayscale frames
  mean=.5
  if any([i!=j for i,j in zip(frame.shape[:2], (params.hl, params.wl))]):
    frame = misc.imresize(frame, [params.hl, params.wl]) # ndframe.gaussian_filter(misc.imresize(frame, [params.hl, params.wl]), 1)
  frame = frame.astype(np.float32) #uint8 -> float32
  frame = (frame/255 - mean)/mean
  
  return frame

def preprocess_evaluate(images, labels, params=None):
  mean=.5
  proimages = (images - mean)/mean
  prolabels = labels
  
  return proimages, prolabels

def random_color(images, labels, params=None):
  col_r = tf.random_uniform([], minval=0, maxval=5, dtype=tf.int32)
  def maybe_transform(image):
    def func(color_ordering):
      return functools.partial(distort_color, image, color_ordering=color_ordering)
    return tf.case({tf.equal(col_r,0): func(0),
                    tf.equal(col_r,1): func(1),
                    tf.equal(col_r,2): func(2),
                    tf.equal(col_r,3): func(3),
                    tf.equal(col_r,4): lambda : image},
                   default=lambda: image)
  proimages = tf.stack([maybe_transform(im) for im in tf.unstack(images)])
  prolabels = labels
  
  return proimages, prolabels
 
def random_blur(images, labels, params=None):
  blu_r = tf.random_uniform([], minval=0, maxval=3, dtype=tf.int32)
  def maybe_transform(image):
    def func(blur_selector):
      return functools.partial(distort_blur, image, blur_selector=blur_selector)
    return tf.case({tf.equal(blu_r,0): func(0),
                    tf.equal(blu_r,1): func(1),
                    tf.equal(blu_r,2): lambda : image},
                   default=lambda: image)
  proimages = tf.stack([maybe_transform(im) for im in tf.unstack(images)])
  prolabels = labels
  
  return tf.clip_by_value(proimages, 0.0, 1.0), prolabels
  
def distort_blur(image, blur_selector=0, scope=None):
  # blur image with selected type of blur
  # image: 3D, float32, in [0,1), numpy array
  assert 0<=blur_selector<=1, 'blur_selector outside of bounds.'
  def blur_function(img, blur_selector):
    # img: 3D numpy array
    # blur_selector in [0,1]
    #print(img.shape, img.dtype)
    random_int = 2*np.random.randint(0, 4) + 1
    if blur_selector==0:
      #print('median')
      # median blur asks the input to have specific type
      img = (img*255).astype(np.uint8)
      return cv2.medianBlur(img,random_int).astype(np.float32)/255
    elif blur_selector==1:
      #print('bilateral')
      # 75,75: good for 2MP, 35,35: 0.5 MP
      return cv2.bilateralFilter(img,random_int,35,35)
  blurred = tf.py_func(blur_function, [image, blur_selector], tf.float32, stateful=True)
  blurred.set_shape(image.get_shape())
  # print('debug:blurred:', blurred.get_shape().as_list())
  return blurred

  
## copied from https://github.com/tensorflow/models/blob/master/slim/preprocessing/inception_preprocessing.py

'''
def apply_with_random_selector(x, func, num_cases):
  """Computes func(x, sel), with sel sampled from [0...num_cases-1].

  Args:
    x: input Tensor.
    func: Python function to apply.
    num_cases: Python int32, number of cases to sample sel from.

  Returns:
    The result of func(x, sel), where func receives the value of the
    selector as a python integer, but sel is sampled dynamically.
  """
  sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
  # Pass the real x only to one of the func calls.
  return control_flow_ops.merge([
      func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
      for case in range(num_cases)])[0]
'''

def distort_color(image, color_ordering=0, scope=None):
  """Distort the color of a Tensor image.

  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.

  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    max_delta_brightness = 55.0/255 #32. / 255.
    lower_contrast = 0.25 #0.5
    upper_contrast = 2.5 #1.5
    lower_saturation = 0.25 #0.5
    upper_saturation = 2.0 #1.5
    if color_ordering == 0:
      image = tf.image.random_brightness(image, max_delta=max_delta_brightness)
      image = tf.image.random_saturation(image, lower=lower_saturation, upper=upper_saturation)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_contrast(image, lower=lower_contrast, upper=upper_contrast)
    elif color_ordering == 1:
      image = tf.image.random_saturation(image, lower=lower_saturation, upper=upper_saturation)
      image = tf.image.random_brightness(image, max_delta=max_delta_brightness)
      image = tf.image.random_contrast(image, lower=lower_contrast, upper=upper_contrast)
      image = tf.image.random_hue(image, max_delta=0.2)
    elif color_ordering == 2:
      image = tf.image.random_contrast(image, lower=lower_contrast, upper=upper_contrast)
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_brightness(image, max_delta=max_delta_brightness)
      image = tf.image.random_saturation(image, lower=lower_saturation, upper=upper_saturation)
    elif color_ordering == 3:
      image = tf.image.random_hue(image, max_delta=0.2)
      image = tf.image.random_saturation(image, lower=lower_saturation, upper=upper_saturation)
      image = tf.image.random_contrast(image, lower=lower_contrast, upper=upper_contrast)
      image = tf.image.random_brightness(image, max_delta=max_delta_brightness)
    else:
      raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)

'''
def distorted_bounding_box_crop(image,
                                bbox,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  See `tf.image.sample_distorted_bounding_box` for more documentation.

  Args:
    image: 3-D Tensor of image (it will be converted to floats in [0, 1]).
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax]. If num_boxes is 0 then it would use the whole
      image.
    min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
      area of the image must contain at least this fraction of any bounding box
      supplied.
    aspect_ratio_range: An optional list of `floats`. The cropped area of the
      image must have an aspect ratio = width / height within this range.
    area_range: An optional list of `floats`. The cropped area of the image
      must contain a fraction of the supplied image within in this range.
    max_attempts: An optional `int`. Number of attempts at generating a cropped
      region of the image of the specified constraints. After `max_attempts`
      failures, return the entire image.
    scope: Optional scope for name_scope.
  Returns:
    A tuple, a 3-D Tensor cropped_image and the distorted bbox
  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox


def preprocess_for_train(image, height, width, bbox,
                         fast_mode=True,
                         scope=None):
  """Distort one image for training a network.

  Distorting images provides a useful technique for augmenting the data
  set during training in order to make the network invariant to aspects
  of the image that do not effect the label.

  Additionally it would create image_summaries to display the different
  transformations applied to the image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged
      as [ymin, xmin, ymax, xmax].
    fast_mode: Optional boolean, if True avoids slower transformations (i.e.
      bi-cubic resizing, random_hue or random_contrast).
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of distorted image used for training with range [-1, 1].
  """
  with tf.name_scope(scope, 'distort_image', [image, height, width, bbox]):
    if bbox is None:
      bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4])
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox)
    tf.summary.image('image_with_bounding_boxes', image_with_box)

    distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])
    image_with_distorted_box = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0), distorted_bbox)
    tf.summary.image('images_with_distorted_bounding_box',
                     image_with_distorted_box)

    # This resizing operation may distort the images because the aspect
    # ratio is not respected. We select a resize method in a round robin
    # fashion based on the thread number.
    # Note that ResizeMethod contains 4 enumerated resizing methods.

    # We select only 1 case for fast_mode bilinear.
    num_resize_cases = 1 if fast_mode else 4
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, method: tf.image.resize_images(x, [height, width], method=method),
        num_cases=num_resize_cases)

    tf.summary.image('cropped_resized_image',
                     tf.expand_dims(distorted_image, 0))

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Randomly distort the colors. There are 4 ways to do it.
    distorted_image = apply_with_random_selector(
        distorted_image,
        lambda x, ordering: distort_color(x, ordering, fast_mode),
        num_cases=4)

    tf.summary.image('final_distorted_image',
                     tf.expand_dims(distorted_image, 0))
    distorted_image = tf.subtract(distorted_image, 0.5)
    distorted_image = tf.multiply(distorted_image, 2.0)
    return distorted_image


def preprocess_for_eval(image, height, width,
                        central_fraction=0.875, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details)
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.
    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

    if height and width:
      # Resize the image to the specified height and width.
      image = tf.expand_dims(image, 0)
      image = tf.image.resize_bilinear(image, [height, width],
                                       align_corners=False)
      image = tf.squeeze(image, [0])
    image = tf.subtract(image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


def preprocess_image(image, height, width,
                     is_training=False,
                     bbox=None,
                     fast_mode=True):
  """Pre-process one image for training or evaluation.

  Args:
    image: 3-D Tensor [height, width, channels] with the image.
    height: integer, image expected height.
    width: integer, image expected width.
    is_training: Boolean. If true it would transform an image for train,
      otherwise it would transform it for evaluation.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
    fast_mode: Optional boolean, if True avoids slower transformations.

  Returns:
    3-D float Tensor containing an appropriately scaled image

  Raises:
    ValueError: if user does not provide bounding box
  """
  if is_training:
    return preprocess_for_train(image, height, width, bbox, fast_mode)
  else:
    return preprocess_for_eval(image, height, width)
  
'''
