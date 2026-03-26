import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np



class Dummies(Layer):
    """
    Layer that creates country and time dummies.
    """

    def __init__(self, N, T, time_periods_na, country_trends=False, within_transform=False, **kwargs):
        super(Dummies, self).__init__(**kwargs)
        self.N = N
        self.T = T
        self.time_periods_na = time_periods_na
        self.noObs = None
        self.country_trends=country_trends
        self.within_transform = within_transform

    def call(self, x):
    
    
        where_mat = tf.transpose(tf.math.is_nan(x))

        for t in range(self.T):
            idx = tf.where(~where_mat[:, t, 0])
            idx = tf.reshape(idx, (-1,))

            #Constructs and identity matrix, so that all count
            D_t = tf.eye(self.N)
            #gathers from D_t according to the indices in idx
            D_t = tf.gather(D_t, idx, axis=0)

            if t == 0:
                Delta_1 = D_t

                Delta_2 = tf.matmul(D_t, tf.ones((self.N, 1)))

            else:
                Delta_1 = tf.concat([Delta_1, D_t], axis=0)

                Delta_2 = tf.concat([Delta_2, tf.zeros((tf.shape(Delta_2)[0], 1))], axis=1)

                Delta_2_tmp = tf.matmul(D_t, tf.ones((self.N, 1)))
                Delta_2_tmp = tf.concat([tf.zeros((tf.shape(Delta_2_tmp)[0], t)), Delta_2_tmp], axis=1)

                Delta_2 = tf.concat([Delta_2, Delta_2_tmp], axis=0)

        Delta_1 = Delta_1[:, 1:]
        Delta_2 = Delta_2[:, self.time_periods_na + 1:]
        
        self.noObs = tf.shape(Delta_1)[0]

        Delta_1 = tf.reshape(Delta_1, (1, self.noObs, self.N - 1))
        Delta_2 = tf.reshape(Delta_2, (1, self.noObs, self.T - (self.time_periods_na + 1)))
            

        return [Delta_1, Delta_2]

  