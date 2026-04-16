import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer


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
        self.country_trends = country_trends
        self.within_transform = within_transform

    def call(self, x):
        where_mat = tf.transpose(tf.math.is_nan(x))

        for t in range(self.T):
            idx = tf.where(~where_mat[:, t, 0])
            idx = tf.reshape(idx, (-1,))

            D_t = tf.eye(self.N)
            D_t = tf.gather(D_t, idx, axis=0)

            if t == 0:
                Delta_1 = D_t
                Delta_2 = tf.matmul(D_t, tf.ones((self.N, 1), dtype=D_t.dtype))
            else:
                Delta_1 = tf.concat([Delta_1, D_t], axis=0)

                Delta_2 = tf.concat(
                    [Delta_2, tf.zeros((tf.shape(Delta_2)[0], 1), dtype=Delta_2.dtype)],
                    axis=1
                )

                Delta_2_tmp = tf.matmul(D_t, tf.ones((self.N, 1), dtype=D_t.dtype))
                Delta_2_tmp = tf.concat(
                    [tf.zeros((tf.shape(Delta_2_tmp)[0], t), dtype=Delta_2_tmp.dtype), Delta_2_tmp],
                    axis=1
                )

                Delta_2 = tf.concat([Delta_2, Delta_2_tmp], axis=0)

        Delta_1 = Delta_1[:, 1:]
        Delta_2 = Delta_2[:, self.time_periods_na + 1:]

        self.noObs = tf.shape(Delta_1)[0]

        Delta_1 = tf.reshape(Delta_1, (1, self.noObs, self.N - 1))
        Delta_2 = tf.reshape(Delta_2, (1, self.noObs, self.T - (self.time_periods_na + 1)))

        return [Delta_1, Delta_2]


class CountryTimeTrends(Layer):
    """
    Build country-specific linear and quadratic time trend design matrices
    from reduced country and time dummy matrices.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        Delta_1, Delta_2 = inputs

        ref_col = 1.0 - tf.reduce_sum(Delta_1, axis=-1, keepdims=True)
        Delta_full = tf.concat([ref_col, Delta_1], axis=-1)  # (1, noObs, N)

        sum_d2 = tf.reduce_sum(Delta_2, axis=-1, keepdims=True)
        time_onehots = tf.concat([1.0 - sum_d2, Delta_2], axis=-1)  # (1, noObs, T_kept)

        time_positions = tf.cast(tf.range(tf.shape(time_onehots)[-1]), dtype=Delta_1.dtype)
        time_positions = tf.reshape(time_positions, (1, 1, -1))

        time_idx = tf.reduce_sum(time_onehots * time_positions, axis=-1, keepdims=True)
        time_idx_sq = tf.square(time_idx)

        linear_all = Delta_full * time_idx
        quad_all = Delta_full * time_idx_sq

        return linear_all, quad_all