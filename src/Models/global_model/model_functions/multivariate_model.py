import numpy as np
import tensorflow as tf
import pandas as pd 
import os 

from .helper_functions import initialize_parameters, Preprocess, individual_loss, WithinHelper
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
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
        
        self.node = node
        self.x_train = x_train
        self.y_train = y_train
        self.x_train_val = x_train_val
        self.y_train_val = y_train_val
        self.y_val = y_val
        self.x_val = x_val
        self._cache = {}
        
        for key, value in dict(cfg).items():
            setattr(self, key, value)
        
        

    def _model_definition(self):
        
        # Initializing parameters
        initialize_parameters(self)
        
        # # Preprocessing data - for both precipitation and temperature        
        Preprocess(self)
        
        #note we have 2 observations for each country, one for precipitation and one for temperature, therefore the input is of dimension (T, N, 2)
        SetupGlobalModel(self)
        
    def get_model(self):
        tf.keras.backend.clear_session()
    
        # # Build model fresh
        # self._model_definition()
        # return self
        
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
            if self.data_source=='wb':
                P_helper_full=WithinHelper(self.input_data[self.input_vars[0]])
                P_matrix_full=P_helper_full.calculate_P_matrix()
                p_tensor=tf.convert_to_tensor(P_matrix_full, dtype=tf.float32)

                y_train=tf.matmul(p_tensor, tf.cast(tf.reshape(self.targets[~self.Mask], (1, -1, 1)), dtype=tf.float32))[:, :self.noObs['train'], :]
                
                y_val=tf.matmul(p_tensor, tf.cast(tf.reshape(self.targets[~self.Mask], (1, -1, 1)), dtype=tf.float32))[:, self.noObs['train']:, :]

                n_obs_holdout= self.noObs['global'] - self.noObs['train']
            
                self.model.compile(optimizer=Adam(lr), loss=individual_loss(mask=self.Mask, p_matrix=p_tensor, n_holdout=n_obs_holdout, balanced=False))
            
            elif self.data_source=='ee':
                y_obs = tf.cast(tf.reshape(self.targets[~self.Mask], (1, -1, 1)), tf.float32)

                T = tf.shape(self.targets)[1].numpy()  # if your stored targets are (batch, T, N) you already know T,N
                N = tf.shape(self.targets)[2].numpy()

                # For balanced: self.Mask is all False, so y_obs length = T*N
                y_within = WithinHelper.within_twfe_balanced(y_obs, T, N)

                y_train = y_within[:, :self.noObs["train"], :]
                y_val   = y_within[:, self.noObs["train"]:, :]
                
                n_obs_holdout= self.noObs['global'] - self.noObs['train']
            
                self.model.compile(optimizer=Adam(lr), loss=individual_loss(mask=self.Mask, p_matrix=None, n_holdout=n_obs_holdout, balanced=True))
            

            callbacks = [EarlyStopping(monitor='val_loss', mode='min', min_delta=min_delta, patience=patience,
                                restore_best_weights=True, verbose=verbose)
            ]
            
            
            #validation data preprocessing
            x_train = [self.input_data_train[var] for var in self.input_vars]
            x_val = [self.input_data_val[var] for var in self.input_vars]
            

            self.model.fit(x_train, y_train, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False, validation_data=(x_val, y_val))
            self.holdout_loss = np.min(self.model.history.history['val_loss'])
            
            del p_tensor
        else:
           
            y_train= tf.reshape(self.targets[~self.Mask], (1, -1, 1))
            self.model.compile(optimizer=Adam(lr), loss=individual_loss(mask=self.Mask, p_matrix=None, n_holdout=0, balanced=False))


            callbacks = [EarlyStopping(monitor='loss', mode='min', min_delta=min_delta, patience=patience,
                                   restore_best_weights=True, verbose=verbose)
                
                     ]

            x_train = [self.input_data[var] for var in self.input_vars]
            
         
            
            self.model.fit(x_train, y_train, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)
 
    
        self.best_weights = self.model.get_weights()
        self.epochs = self.model.history.epoch
        

        #fixed effects
        # self.alpha = pd.DataFrame(self.country_FE_layer.weights[0].numpy().T)
        # self.alpha.columns = self.individuals['global'][1:196]
        # self.beta = pd.DataFrame(self.time_FE_layer.weights[0].numpy())
        # self.beta.set_index(self.time_periods[self.time_periods_not_na['global']][1:196], inplace=True)
        
    def load_params(self, filepath):
        """
        Loading model parameters.

         ARGUMENTS
            * filepath: string containing path/name of saved file.
        """

        self.model.load_weights(filepath)
        self.params = self.model.get_weights()
        if self.holdout==   0:
            # loading fixed effects estimates
            self.alpha = pd.DataFrame(self.country_FE_layer.weights[0].numpy().T)
            self.alpha.columns = self.individuals['global'][1:self.N['global']]
            self.beta = pd.DataFrame(self.time_FE_layer.weights[0].numpy())
            self.beta.set_index(self.time_periods[self.time_periods_not_na['global']][1:self.N['global']], inplace=True)

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