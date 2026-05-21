import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class Extend(Layer):
    """
    Layer that extends the second dimension of inputs.
    """

    def __init__(self, mask, **kwargs):
        super(Extend, self).__init__(**kwargs)
        self.mask = mask

    def call(self, x):
        where = ~self.mask
        indices = tf.cast(tf.where(where), tf.int32)
        scatter = tf.scatter_nd(indices, tf.reshape(x, (-1,)), shape=tf.shape(self.mask))
        scatter = tf.cast(scatter, dtype=np.float64)

        indices = tf.cast(tf.where(~where), tf.int32)
        mask_tmp = tf.cast(self.mask, tf.int32)
        x_nan = tf.ones(tf.reduce_sum(mask_tmp)) * np.nan
        scatter_nan = tf.scatter_nd(indices, x_nan, shape=tf.shape(self.mask))
        scatter_nan = tf.cast(scatter_nan, dtype=np.float64)

        return scatter + scatter_nan

