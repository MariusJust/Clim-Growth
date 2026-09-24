import numpy as np
import tensorflow as tf
import pandas as pd
import os
from models.helper_functions.regional_model import initialize_parameters, Preprocess, individual_loss, WithinHelper
from tensorflow.keras.optimizers import Adam
from models.helper_functions.shared import build_optimizer
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from .model_architecture_reg import Regions

class MultivariateModel:
    """
    Class implementing the static neural network model.
    """

    def __init__(self, node, cfg, x_train=None, y_train=None, x_train_val=None, y_train_val=None, y_val=None, x_val=None):
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
        self.region_builders=[]
        self.country_map = None

        # Compatibility: ensure holdout exists and is zeroed (holdout path removed)
        self.holdout = int(getattr(cfg, "holdout", 0) or 0)

        #unpack config
        for key, value in dict(cfg).items():
            setattr(self, key, value)

        self.node = node



    def _model_definition(self):

        initialize_parameters(self)

        #  Preprocessing data - for both precipitation and temperature
        Preprocess(self)

        # Create model instance for each region
        Regions(self, regions=self.regions).SetupRegionalModel()


    def get_model(self):

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

        if self.holdout>0:
            #compute p matrix on whole data
            P_helper_regional=WithinHelper(self.input_data_temp)
            P_list=P_helper_regional.calculate_P_matrix()
            p_tensor=[tf.convert_to_tensor(P_list[i], dtype=tf.float32) for i in range(len(P_list))]

            y_true_target_train=[tf.matmul(p_tensor[i], tf.cast(tf.reshape(self.targets[i][~self.masks[i]], (1, -1, 1)), dtype=tf.float32))[:, :self.noObs["train"][region], :] for i, region in enumerate(self.regions)]

            y_true_target_val=[tf.matmul(p_tensor[i], tf.cast(tf.reshape(self.targets[i][~self.masks[i]], (1, -1, 1)), dtype=tf.float32))[:, self.noObs["train"][region]:, :] for i, region in enumerate(self.regions)]

            n_obs_holdout= [self.noObs[region] - self.noObs["train"][region] for region in self.regions]

            self.model.compile(optimizer=build_optimizer(getattr(self, "optimizer", "adam"), lr, cfg=self), loss=[individual_loss(mask=self.masks[i], p_matrix=p_tensor[i], n_holdout=n_obs_holdout[i], name=f"loss_{region}") for i, region in enumerate(self.regions)], loss_weights=[1 / self.no_regions] * self.no_regions)


            callbacks = [EarlyStopping(monitor='val_loss', mode='min', min_delta=min_delta, patience=patience,
                                restore_best_weights=True, verbose=verbose)
            ]
            if bool(getattr(self, "reduce_lr_on_plateau", False)):
                callbacks.append(ReduceLROnPlateau(
                    monitor='val_loss', mode='min',
                    factor=float(getattr(self, "lr_reduce_factor", 0.3)),
                    patience=int(getattr(self, "lr_reduce_patience", 50)),
                    min_delta=float(min_delta),
                    min_lr=float(getattr(self, "lr_min", 1.0e-7)),
                    verbose=verbose,
                ))


            #validation data preprocessing
            x_train_val = [self.input_data_temp_train, self.input_data_precip_train]
            x_val = [self.input_data_temp_val, self.input_data_precip_val]


            self.model.fit(x_train_val, y_true_target_train, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False, validation_data=(x_val, y_true_target_val))
            self.holdout_loss = np.min(self.model.history.history['val_loss'])

            del p_tensor
        else:
            self.model.compile(optimizer=build_optimizer(getattr(self, "optimizer", "adam"), lr, cfg=self), loss=self.loss_list, loss_weights=[1 / self.no_regions] * self.no_regions)


            callbacks = [EarlyStopping(monitor='loss', mode='min', min_delta=min_delta, patience=patience,
                                    restore_best_weights=True, verbose=verbose)

                        ]
            if bool(getattr(self, "reduce_lr_on_plateau", False)):
                callbacks.append(ReduceLROnPlateau(
                    monitor='loss', mode='min',
                    factor=float(getattr(self, "lr_reduce_factor", 0.3)),
                    patience=int(getattr(self, "lr_reduce_patience", 50)),
                    min_delta=float(min_delta),
                    min_lr=float(getattr(self, "lr_min", 1.0e-7)),
                    verbose=verbose,
                ))

            x_train = [self.input_data_temp, self.input_data_precip]
            self.model.fit(x_train, self.targets, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)


        self.best_weights = self.model.get_weights()
        self.epochs = self.model.history.epoch


        #saving fixed effects
        if self.holdout==0 and not bool(getattr(self, "within_projection", False)):
            for i, region in enumerate(self.regions):
                self.alpha[region] = pd.DataFrame(self.country_FE_layer[i].weights[0].numpy().T)
                self.alpha[region].columns = self.individuals[region][1:]

                time_vals = self.time_FE_layer[i].weights[0].numpy().flatten()
                time_names = self.time_periods[self.time_periods_na[region] + 1:]
                self.beta[region] = pd.DataFrame({"time": time_names, "estimate": time_vals}).set_index("time")

            if bool(getattr(self, "country_trends", False)):
                use_quadratic = bool(getattr(self, "quadratic_trends", True))
                self.linear_trend = {}
                self.quadratic_trend = {} if use_quadratic else None
                trend_groups = getattr(self, "trend_groups", None)
                for i, region in enumerate(self.regions):
                    linear_vals = self.linear_trend_layer[i].weights[0].numpy().flatten()
                    # Grouped trends -> one weight per country; otherwise per fid.
                    if trend_groups is not None and trend_groups.get(region) is not None:
                        all_countries = list(trend_groups[region])
                    else:
                        all_countries = list(self.individuals[region])

                    self.linear_trend[region] = pd.DataFrame({"country": all_countries, "estimate": linear_vals}).set_index("country")
                    if use_quadratic:
                        quad_vals = self.quadratic_trend_layer[i].weights[0].numpy().flatten()
                        self.quadratic_trend[region] = pd.DataFrame({"country": all_countries, "estimate": quad_vals}).set_index("country")


    def load_params(self, filepath):
        """
        Loading model parameters.

         ARGUMENTS
            * filepath: string containing path/name of saved file.
        """

        try:
            self.model.load_weights(filepath)
        except ValueError:
            self.model.load_weights(filepath, skip_mismatch=True)
        self.params = self.model.get_weights()

        # In within mode the FE/trend layers are concentrated out (None), so
        # there are no FE summaries to rebuild; model_visual[region] (the climate
        # net) is still available for surfaces.
        if not bool(getattr(self, "within_projection", False)):
            for i, region in enumerate(self.regions):
                self.alpha[self.regions[i]] = pd.DataFrame(self.country_FE_layer[i].weights[0].numpy().T)
                self.alpha[self.regions[i]].columns = self.individuals[self.regions[i]][1:]

                self.beta[self.regions[i]] = pd.DataFrame(self.time_FE_layer[i].weights[0].numpy())
                self.beta[self.regions[i]].set_index(self.time_periods[self.time_periods_na[self.regions[i]] + 1:], inplace=True)


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
        in_sample_preds = self.model([self.input_data_temp, self.input_data_precip])
        within = bool(getattr(self, "within_projection", False))
        noObs_tmp = 0
        MSE = 0
        rank_sum = 0

        for region in self.regions:
                idx = self.regions.index(region)
                self.in_sample_pred[region] = self.y_train[region].copy()
                self.in_sample_pred[region].iloc[:, :] = np.array(in_sample_preds[idx][0, :, :])

                noObs_tmp = noObs_tmp + self.noObs[region]

                if within:
                    # Within mode: output is the climate net; recover FE/trends by
                    # exact OLS per region and report the FULL R2/SSE.
                    proj = self.within_projs[idx]
                    y_mat = np.array(self.y_train_df[region], dtype=float)
                    f_mat = np.array(self.in_sample_pred[region], dtype=float)
                    y_obs = y_mat[proj.t_arr, proj.n_arr]
                    f_obs = f_mat[proj.t_arr, proj.n_arr]
                    gamma = proj.recover_gamma(y_obs - f_obs)
                    full_obs = f_obs + proj.W @ gamma
                    SSE = float(np.sum((y_obs - full_obs) ** 2))
                    mean_tmp = float(np.nanmean(y_mat))
                    SST = float(np.sum((y_obs - mean_tmp) ** 2))
                    rank_sum += int(proj.rank)
                else:
                    mean_tmp = np.nanmean(np.array(self.y_train_df[region]))
                    SSE = np.nansum(np.nansum((self.y_train_df[region] - self.in_sample_pred[region]) ** 2))
                    SST = np.nansum(np.nansum((self.y_train_df[region] - mean_tmp) ** 2))

                self.R2[region] = 1 - SSE / SST if SST > 0 else np.nan
                MSE = MSE + SSE / self.noObs[region]

        m_eff = int(self.m) + rank_sum if within else self.m
        self.m_effective = m_eff
        self.BIC = np.log(MSE)*noObs_tmp + m_eff * np.log(noObs_tmp)
        self.AIC = np.log(MSE)*noObs_tmp + 2 * m_eff

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
