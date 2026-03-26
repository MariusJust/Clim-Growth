import numpy as np
import tensorflow as tf
import pandas as pd 
import os 
from .helper_functions import initialize_parameters, Preprocess, individual_loss,  WithinHelper
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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

        self.node = node
        self.x_train = x_train
        self.y_train = y_train
        self.x_train_val = x_train_val
        self.y_train_val = y_train_val
        self.y_val = y_val
        self.x_val = x_val
        self._cache = {}
        self.region_builders=[]

        #unpack config
        for key, value in dict(cfg).items():
            setattr(self, key, value)
            
        
    
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
        
            self.model.compile(optimizer=Adam(lr), loss=[individual_loss(mask=self.masks[i], p_matrix=p_tensor[i], n_holdout=n_obs_holdout[i], name=f"loss_{region}") for i, region in enumerate(self.regions)], loss_weights=[1 / self.no_regions] * self.no_regions)
            

            callbacks = [EarlyStopping(monitor='val_loss', mode='min', min_delta=min_delta, patience=patience,
                                restore_best_weights=True, verbose=verbose)
            ]
            
            
            #validation data preprocessing
            x_train_val = [self.input_data_temp_train, self.input_data_precip_train]
            x_val = [self.input_data_temp_val, self.input_data_precip_val]
            

            
            self.model.fit(x_train_val, y_true_target_train, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False, validation_data=(x_val, y_true_target_val))
            self.holdout_loss = np.min(self.model.history.history['val_loss']) 
            
            del p_tensor
        else:
            self.model.compile(optimizer=Adam(lr), loss=self.loss_list, loss_weights=[1 / self.no_regions] * self.no_regions)


            callbacks = [EarlyStopping(monitor='loss', mode='min', min_delta=min_delta, patience=patience,
                                    restore_best_weights=True, verbose=verbose)
                
                        ]

            x_train = [self.input_data_temp, self.input_data_precip]
            self.model.fit(x_train, self.targets, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)

    
        self.best_weights = self.model.get_weights()
        self.epochs = self.model.history.epoch
        
        
        #saving fixed effects
        if self.holdout==0:
            for i in range(self.no_regions):
                    self.alpha[self.regions[i]] = pd.DataFrame(self.country_FE_layer[i].weights[0].numpy().T)
                    self.alpha[self.regions[i]].columns = self.individuals[self.regions[i]][1:]

        
    def load_params(self, filepath):
        """
        Loading model parameters.

         ARGUMENTS
            * filepath: string containing path/name of saved file.
        """

        self.model.load_weights(filepath)
        self.params = self.model.get_weights()
      
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
        noObs_tmp = 0
        MSE = 0
        
        for region in self.regions:
                self.in_sample_pred[region] = self.y_train[region].copy()
                self.in_sample_pred[region].iloc[:, :] = np.array(in_sample_preds[self.regions.index(region)][0, :, :])

                if self.regions.index(region) == 0:
                    in_sample_pred_global = self.in_sample_pred[region]
                    in_sample_global = self.y_train_df[region]
                else:
                    in_sample_pred_global = pd.concat([in_sample_pred_global, self.in_sample_pred[region]], axis=1)
                    in_sample_global = pd.concat([in_sample_global, self.y_train_df[region]], axis=1)
                    
                noObs_tmp = noObs_tmp + self.noObs[region]
                mean_growth = np.nanmean(np.reshape(np.array(in_sample_global), (-1)))
        
                SST = np.nansum((in_sample_global- mean_growth) ** 2)
                SSR = np.nansum((in_sample_pred_global - mean_growth) ** 2)
                SSE = np.nansum((in_sample_global - in_sample_pred_global) ** 2)

                self.R2[region] = SSR / SST
                MSE = MSE + SSE / self.noObs[region]

   
        self.BIC = np.log(MSE)*noObs_tmp + self.m * np.log(noObs_tmp)
        self.AIC = np.log(MSE)*noObs_tmp + 2 * self.m
        
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