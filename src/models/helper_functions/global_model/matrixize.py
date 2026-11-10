import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class Matrixize(Layer):
    def __init__(self, N, T, mask, holdout, n_obs_train, **kwargs):
        super(Matrixize, self).__init__(**kwargs)
        self.N = N
        self.T = T
        self.mask = mask
        self.holdout = holdout
        self.noObs_train = n_obs_train
    def call(self, x):

        # tf.print(">>> Matrixize debug: t initial:", self.T)
        #when we get the input x, we can retrieve the dimenionsion of x
        n_obs=tf.shape(x)[1]
        
        def val_branch():

           mask = self.mask[:, -self.holdout:, :]
           where = tf.math.logical_not(mask)
           noObs = tf.reduce_sum(tf.cast(where, tf.int32))
           T=self.holdout
           return mask, where, noObs, T

        def train_branch():
            if self.holdout>0:
                mask = self.mask[:, :-self.holdout, :]
            else:
                mask = self.mask
                
            where = tf.math.logical_not(mask)
            noObs = tf.reduce_sum(tf.cast(where, tf.int32))
            T=self.T

            return mask, where, noObs, T

        chosen_mask, where, noObs, T = tf.cond(tf.equal(n_obs, self.noObs_train), train_branch, val_branch)

        out_shape = tf.shape(chosen_mask)
    
        indices = tf.cast(tf.where(where), tf.int32)

        updates_obs = tf.reshape(x, (-1,))

        scatter = tf.scatter_nd(indices, updates_obs, shape=out_shape)
        scatter = tf.cast(scatter, dtype=np.float64)
        
        indices = tf.cast(tf.where(~where), tf.int32)

        # tf.print(">>>self.T:", T)
        x_nan = tf.ones(self.N * T - noObs) * np.nan
        scatter_nan = tf.scatter_nd(indices, x_nan, shape=out_shape)
        scatter_nan = tf.cast(scatter_nan, dtype=np.float64)

        return scatter + scatter_nan