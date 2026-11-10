
import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

# %% Creating vectorization layer
class Vectorize(Layer):
    """
    Layer that vectorizes the second dimension of inputs.

    Time scaling
    ------------
    In the dynamic model (Bennedsen, Hillebrand & Jensen 2022, eq. 4.9) time
    enters the first hidden layer as ``kappa + Gamma_x x_it + Gamma_t t``.
    Temperature and precipitation reach this layer already standardised, but the
    raw time index runs 1..T (T = 63 here), so with ``glorot_uniform``
    initialisation (limit sqrt(6/(fan_in+fan_out)) ~ 1.1) the time term supplies
    roughly 40x more of the pre-activation than the climate inputs. The result is
    saturation at initialisation: sd(z) ~ 23, and about 46% of units land where
    swish'(z) < 0.01 and therefore receive no gradient (swish'(-25) = -3e-10).
    In the 2026-07-02 run this killed the temperature-carrying unit -- its
    pre-activation falls from -0.6 in 1961 to -25.2 in 2020 -- so the estimated
    climate response collapsed from 4.9 pp to ~1 pp of range.

    Rescaling time is an exact reparameterisation, NOT a change of model: with
    ``t~ = (t - mu) / sigma`` the identical pre-activation is reproduced by
    ``Gamma_t~ = sigma * Gamma_t`` and ``kappa~ = kappa + mu * Gamma_t``. The
    representable function class is unchanged; only the conditioning of the
    optimisation problem improves.

    ``time_scaling``:
      ``'standardize'`` (default)  t~ = (t - (T+1)/2) / sqrt((T^2-1)/12).
          The moments are those of the deterministic sequence 1..T, so the
          transform does not depend on the missingness pattern or on which
          countries a bootstrap replicate happens to draw. Reproducible and
          identical across replicates.
      ``'none'``  legacy behaviour: raw 1..T. Kept so earlier runs (e.g.
          2026-07-02_12-01-08_global_dynamic) remain reproducible.

    ``self.time_mu`` / ``self.time_sigma`` record the constants so downstream
    code can map an estimated time effect back to calendar time.
    """

    def __init__(self, N, variable, time_periods=None, time_scaling="standardize",
                 **kwargs):
        super(Vectorize, self).__init__(**kwargs)
        self.N = N
        self.time_periods = time_periods
        self.time_scaling = str(time_scaling).lower()

        self.dim1 = None
        self.variable = variable

        # Deterministic moments of the 1..T index, resolved at construction.
        self.time_mu, self.time_sigma = 0.0, 1.0
        if variable == "time" and time_periods is not None:
            T = int(np.asarray(time_periods).shape[0])
            if self.time_scaling in ("standardize", "standardise", "z"):
                self.time_mu = (T + 1.0) / 2.0
                self.time_sigma = float(np.sqrt(max(T * T - 1.0, 1.0) / 12.0))
            elif self.time_scaling in ("none", "raw", "off"):
                pass
            else:
                raise ValueError(
                    f"unknown time_scaling {time_scaling!r}; "
                    "use 'standardize' or 'none'")

    def call(self, x):
        mask = tf.math.is_nan(x)

        if self.variable == 'time':
            time_periods = tf.reshape(self.time_periods, (1, -1, 1)) - (self.time_periods[0] - 1)
            time_mat = tf.repeat(time_periods, repeats=self.N, axis=2)
            var_clean = tf.reshape(tf.cast(time_mat[~mask], dtype=np.float32), (1, -1, 1))
            if self.time_sigma != 1.0 or self.time_mu != 0.0:
                var_clean = (var_clean - tf.cast(self.time_mu, var_clean.dtype)) \
                            / tf.cast(self.time_sigma, var_clean.dtype)

        else:

            # Apply the mask to both slices.
            var_clean = tf.reshape(x[~mask], (1, -1, 1))

            # Store the shape of the first variable for further processing
            self.dim1 = tf.shape(var_clean)[1]

        return var_clean


    def compute_output_shape(self):
        return (1, self.dim1, 1)

    def get_config(self):
        cfg = super().get_config()
        cfg.update(dict(N=self.N, variable=self.variable,
                        time_scaling=self.time_scaling))
        return cfg
