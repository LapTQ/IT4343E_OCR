import tensorflow as tf
from tensorflow import keras
from keras.engine import base_layer, base_preprocessing_layer
import numpy as np

H_AXIS = -3
W_AXIS = -2
C_AXIS = -1

class Resize(base_layer.Layer):
    def __init__(self, height, width, mode='median', **kwargs):
        """
        Crop to to the target shape and keep aspect ratio.
        Padding image uses the median.
        :param height: target height
        :param width: target shape
        """
        self.height = height
        self.width = width
        self.mode = mode
        super(Resize, self).__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell('Resize').set(True)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[H_AXIS] = self.height
        input_shape[W_AXIS] = self.width
        return tf.TensorShape(input_shape)

    def get_config(self):
        config = super(Resize, self).get_config()
        config.update(
            {
                'height': self.height,
                'width': self.width,
                'mode': self.mode
            }
        )
        return config

    def call(self, inputs):
        H, W = self.height, int(inputs.shape[W_AXIS] * self.height / inputs.shape[H_AXIS])
        outputs = tf.image.resize(inputs, (H, W))
        outputs = np.pad(outputs, ((0, 0), (0, self.width - W), (0, 0)), mode=self.mode)
        return outputs

class RGB2Gray(base_layer.Layer):
    """
    Converts one or more images from RGB to Grayscale.
    The size of the last dimension of the output is 1.
    """
    def __init__(self, **kwargs):
        super(RGB2Gray, self).__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell('RGB2Gray').set(True)

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape).as_list()
        input_shape[C_AXIS] = 1
        return tf.TensorShape(input_shape)

    def call(self, inputs):
        outputs = tf.image.rgb_to_grayscale(inputs)
        return outputs




