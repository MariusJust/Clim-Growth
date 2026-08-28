import numpy as np
import tensorflow as tf
import pandas as pd
import os

from models.helper_functions.global_model import initialize_parameters, Preprocess, individual_loss, WithinHelper
from models.helper_functions.shared import build_optimizer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .model_architecture import SetupGlobalModel


class MultivariateModel:
    """
    Class implementing the static neural network model.
    """

    def __init__(self, node, cfg,  x_train=None, y_train=None, x_train_val=None, y_train_val=None, y_val=None, x_val=None):
        """
        Instantiating class.

        ARGUMENT
            * node:          tuple defining the model architecture.
            * x_train:        array of dicts of TxN_r dataframes of input data (aligned) with a key for each region.
            * y_train:        dict of TxN_r dataframes of target data (aligned) with a key for each region.
            * formulation:    str determining the formulation of the model. Must be one of 'global' or 'regional' or 'national'.

        NB: regions are inferred from the    keys of x_train and y_train.
        """

        self.x_train = x_train
        self.y_train = y_train
        self.x_train_val = x_train_val
        self.y_train_val = y_train_val
        self.y_val = y_val
        self.x_val = x_val
        self._cache = {}
        self.country_map = None
        self.holdout = int(getattr(cfg, "holdout", 0) or 0)

        for key, value in dict(cfg).items():
            setattr(self, key, value)

        self.node = node


    def _model_definition(self):

        # Initializing parameters
        initialize_parameters(self)

        # # Preprocessing data - for both precipitation and temperature
        Preprocess(self)

        #note we have 2 observations for each country, one for precipitation and one for temperature, therefore the input is of dimension (T, N, 2)
        SetupGlobalModel(self)

    def get_model(self):
        tf.keras.backend.clear_session()

        key = tuple(self.node)
        if key not in self._cache:
            self._model_definition()
            self._cache[key] = self
        return self._cache[key]

    def fit(self, lr, min_delta, patience, verbose):

        """
        Fitting the model.

        ARGUMENTS
            * lr:            initial learning rate of the Adam optimizer.
            * min_delta:     tolerance to be used for optimization.
            * patience:      patience to be used for optimization.
            * verbose:       verbosity mode for optimization.
        """

        y_train = tf.reshape(self.targets[~self.Mask], (1, -1, 1))
        optimizer = build_optimizer(getattr(self, "optimizer", "adam"), lr, cfg=self)
        if bool(getattr(self, "within_projection", False)):
            proj = self._get_within_projector()
            y_mat_t = np.array(self.y_train_transf['global'], dtype=np.float64)
            y_obs_t = y_mat_t[proj.t_arr, proj.n_arr]
            within = dict(
                W=tf.constant(proj.W, dtype=tf.float32),
                B=tf.constant(proj.B, dtype=tf.float32),
                Py=tf.constant(proj.annihilate(y_obs_t), dtype=tf.float32),
            )
            loss_fn = individual_loss(mask=self.Mask, within=within)
        else:
            loss_fn = individual_loss(mask=self.Mask, p_matrix=None, n_holdout=0, balanced=False)
        self.model.compile(optimizer=optimizer, loss=loss_fn)

        callbacks = []
        callbacks.append(EarlyStopping(
            monitor='loss', mode='min', min_delta=min_delta, patience=patience,
            restore_best_weights=True, verbose=verbose,
        ))

        if bool(getattr(self, "reduce_lr_on_plateau", False)):
            callbacks.append(ReduceLROnPlateau(
                monitor='loss', mode='min',
                factor=float(getattr(self, "lr_reduce_factor", 0.3)),
                patience=int(getattr(self, "lr_reduce_patience", 50)),
                min_delta=float(min_delta),
                min_lr=float(getattr(self, "lr_min", 1.0e-7)),
                verbose=verbose,
            ))

        x_train = [self.input_data[var] for var in self.input_vars]

        self.model.fit(x_train, y_train, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)

        self.best_weights = self.model.get_weights()
        self.epochs = self.model.history.epoch
        self.refresh_summaries()

    def _within_summaries(self, gamma):
        """Populate FE/trend summaries in within mode from the FWL-recovered
        coefficients gamma = (WᵀW)⁺ Wᵀ (y - f). Blocks follow W's column order:
        [country FE (N) | time FE (T) | country linear (N) | country quad (N)],
        the trend blocks present only when the corresponding handles are on.
        These are the minimum-norm coefficients over the FULL dummy set (no
        reference category dropped), so their level normalisation differs from
        the Dense-layer parameterisation; differences within a block are
        identified, and recentre (e.g. demean) for cross-model comparison.
        """
        N = int(self.N['global']); T = int(self.T)
        g = np.asarray(gamma).flatten()
        countries = list(self.individuals['global'])
        gc = g[0:N]
        self.alpha = pd.DataFrame({"country": countries, "estimate": gc}).set_index("country")
        self.alpha_dict = {c: v for c, v in zip(countries, gc)}
        # The dynamic model drops the time-dummy block from the within projection
        # (time enters the network as an input), so gamma has no time-FE segment.
        has_time = not bool(getattr(self, "dynamic_model", False))
        off = N
        if has_time:
            gt = g[N:N + T]
            try:
                tnames = list(np.asarray(self.time_periods)[self.time_periods_not_na['global']])
            except Exception:
                tnames = list(range(T))
            if len(tnames) != T:
                tnames = list(range(T))
            self.beta = pd.DataFrame({"time": tnames, "estimate": gt}).set_index("time")
            self.beta_dict = {t: v for t, v in zip(tnames, gt)}
            off = N + T
        else:
            self.beta = None
            self.beta_dict = None
        ct = bool(getattr(self, "country_trends", False))
        qt = ct and bool(getattr(self, "quadratic_trends", True))
        if ct:
            glin = g[off:off + N]
            self.linear_trend = pd.DataFrame({"country": countries, "estimate": glin}).set_index("country")
            self.linear_trend_dict = {c: v for c, v in zip(countries, glin)}
            off += N
            if qt:
                gq = g[off:off + N]
                self.quadratic_trend = pd.DataFrame({"country": countries, "estimate": gq}).set_index("country")
                self.quadratic_trend_dict = {c: v for c, v in zip(countries, gq)}
            else:
                self.quadratic_trend = None
                self.quadratic_trend_dict = None
        self.gamma_within = g
        return self

    def _get_within_projector(self):
        """Build/cache the FWL projector (FE + country trends) for within mode."""
        if getattr(self, "within_proj", None) is None:
            from models.helper_functions.global_model.within_projection import WithinProjector
            mask_TN = np.asarray(self.mask['global']).reshape(self.T, self.N['global'])
            ct = bool(getattr(self, "country_trends", False))
            qt = ct and bool(getattr(self, "quadratic_trends", True))
            # Dynamic model carries time as a network input and has no additive time
            # fixed effect, so its within projection annihilates country FE + trends only.
            include_time = not bool(getattr(self, "dynamic_model", False))
            self.within_proj = WithinProjector(mask_TN, country_trends=ct,
                                               quadratic_trends=qt, include_time=include_time)
        return self.within_proj

    def refresh_summaries(self):
        """
        Rebuild fixed-effect and trend summaries from the current layer weights.
        """
        if bool(getattr(self, "within_projection", False)):
            # FE/trends are concentrated out (no Dense layers); recover post-fit
            # via the projector if needed.
            return self

        # country fixed effects
        country_names = list(self.individuals['global'])
        estimated_country_vals = self.country_FE_layer.weights[0].numpy().flatten()

        self.alpha = pd.DataFrame(
            {"country": country_names[1:], "estimate": estimated_country_vals}
        ).set_index("country")

        self.alpha_dict = {
            country: value for country, value in zip(country_names[1:], estimated_country_vals)
        }

        # time fixed effects
        time_names = self.time_periods[self.time_periods_not_na['global']]
        estimated_time_vals = self.time_FE_layer.weights[0].numpy().flatten()

        self.beta = pd.DataFrame(
            {"time": time_names[1:], "estimate": estimated_time_vals}
        ).set_index("time")

        self.beta_dict = {
            time_period: value for time_period, value in zip(time_names[1:], estimated_time_vals)
        }

        if bool(getattr(self, "country_trends", False)):
            use_quadratic = bool(getattr(self, "quadratic_trends", True))
            linear_vals = self.linear_trend_layer.weights[0].numpy().flatten()
            # When trends are grouped by country, the trend layer has one weight
            # per country; otherwise one per fid (self.individuals).
            trend_groups = getattr(self, "trend_groups", None)
            all_countries = list(trend_groups) if trend_groups is not None else list(self.individuals['global'])

            self.linear_trend = pd.DataFrame({"country": all_countries, "estimate": linear_vals}).set_index("country")
            self.linear_trend_dict = {country: value for country, value in zip(all_countries, linear_vals)}

            if use_quadratic:
                quad_vals = self.quadratic_trend_layer.weights[0].numpy().flatten()
                self.quadratic_trend = pd.DataFrame({"country": all_countries, "estimate": quad_vals}).set_index("country")
                self.quadratic_trend_dict = {country: value for country, value in zip(all_countries, quad_vals)}
            else:
                self.quadratic_trend = None
                self.quadratic_trend_dict = None

        return self


    def load_params(self, filepath):
        """
        Loading model parameters.

         ARGUMENTS
            * filepath: string containing path/name of saved file.
        """

        self.model.load_weights(filepath)
        self.params = self.model.get_weights()
        self.refresh_summaries()

    def save_params(self, filepath):
        """
        Saving model parameters.

         ARGUMENTS
            * filepath: string containing path/name of file to be saved.
        """

        self.model.save_weights(filepath)


    def in_sample_predictions(self):
        """
        Making in-sample predictions.

        """


        # Generate in-sample predictions using the model
        in_sample_preds = self.model([self.input_data[var] for var in self.input_vars])

        if bool(getattr(self, "within_projection", False)):
            # Within mode: model output is the climate net only. Recover FE/trends
            # by exact OLS (the FWL profiled fit) and report the FULL R2/SSE/AIC.
            proj = self._get_within_projector()
            f_mat = np.array(in_sample_preds[0, :, 0:self.N['global']], dtype=float)
            self.in_sample_pred['global'] = self.y_train['global'].copy()
            self.in_sample_pred['global'].iloc[:, :] = f_mat            # climate surface only
            y_mat = np.array(self.y_train_df['global'], dtype=float)
            y_obs = y_mat[proj.t_arr, proj.n_arr]
            f_obs = f_mat[proj.t_arr, proj.n_arr]
            gamma = proj.recover_gamma(y_obs - f_obs)                   # exact FE/trends
            full_obs = f_obs + proj.W @ gamma
            self._within_summaries(gamma)
            SSE = float(np.sum((y_obs - full_obs) ** 2))
            mean_tmp = float(np.nanmean(y_mat))
            SST = float(np.sum((y_obs - mean_tmp) ** 2))
            self.R2['global'] = 1 - SSE / SST if SST > 0 else np.nan
            MSE = SSE / self.noObs['global']
            m_eff = int(self.m) + int(proj.rank)
            self.m_effective = m_eff
            self.BIC = np.log(MSE) * self.noObs['global'] + m_eff * np.log(self.noObs['global'])
            self.AIC = np.log(MSE) * self.noObs['global'] + 2 * m_eff
            return in_sample_preds

        # Initialize aggregation variable
        MSE = 0


        # Copy the structure of the observed global data
        self.in_sample_pred['global'] = self.y_train['global'].copy()

        # Replace the copied data with the in-sample predictions
        self.in_sample_pred['global'].iloc[:, :] = np.array(in_sample_preds[0, :, 0:self.N['global']])

        # Flatten the prediction and training data to vectors
        pred_vector = np.reshape(np.array(self.in_sample_pred['global']), (-1))
        train_vector = np.reshape(np.array(self.y_train['global']), (-1))

        # Store the global predictions and actuals for comparison
        in_sample_pred_global = pred_vector
        in_sample_global = train_vector

        SSE = np.nansum((in_sample_global - in_sample_pred_global) ** 2)
        mean_tmp = np.nanmean(np.array(self.y_train_df['global']))
        SST = np.nansum(np.nansum((self.y_train_df['global'] - mean_tmp) ** 2))
        self.R2['global'] = 1 - SSE / SST if SST > 0 else np.nan

        MSE = SSE / self.noObs['global']

        self.BIC = (np.log(MSE))*self.noObs['global'] + self.m * np.log(self.noObs['global'])
        self.AIC= (np.log(MSE))*self.noObs['global'] + 2 * self.m

        return in_sample_preds


    def predict(self, temperature_array, precip_array, idx=False):
        """
        Making predictions.

        ARGUMENTS
            * x_test:  (-1,1) array of input data.
            * idx:     Name identifying the country/region to be used for making predictions (if national or regional formulation).


        RETURNS
            * pred_df: Dataframe containing predictions.
        """

        pred_vector=tf.concat([temperature_array, precip_array], axis=3)

        predictions=self.model_pred.predict(pred_vector)

        return predictions
