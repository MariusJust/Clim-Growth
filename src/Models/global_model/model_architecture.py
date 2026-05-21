from tensorflow.keras.layers import Input, Add, concatenate
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from models.helper_functions.shared import Dummies, CountryTimeTrends, create_fixed_effects, create_country_trends, build_country_grouping, Vectorize, Count_params
from models.helper_functions.global_model import Matrixize, create_hidden_layers, create_output_layer, Visual_model, pred_model

def SetupGlobalModel(self):

    """
    Setting up the global model.
    """

    if self.x_val is not None:
      self.targets_train = tf.reshape(tf.convert_to_tensor(self.y_train_val_transf['global']), (1, self.T, self.N['global']))
      self.targets_val = tf.reshape(tf.convert_to_tensor(self.y_val_transf['global']), (1, self.holdout, self.N['global']))
      self.targets = tf.reshape(tf.convert_to_tensor(self.y_train_transf['global']), (1, self.T + self.holdout, self.N['global']))

      for idx, var in enumerate(self.input_vars):
            self.input[var] = Input(shape=(None, int(self.N['global'])), name=f'{var}_input')
            self.input_data_train[var]= tf.reshape(tf.convert_to_tensor(self.x_train_val_transf[idx]['global']), (1, self.T, self.N['global']))
            self.input_data_val[var] = tf.reshape(tf.convert_to_tensor(self.x_val_transf[idx]['global']), (1, self.holdout, self.N['global']))
            self.input_data[var] = tf.reshape(tf.convert_to_tensor(self.x_train_transf[idx]['global']), (1, self.T + self.holdout, self.N['global']))


    else:
        self.noObs['train'] = self.noObs['global']
        self.targets = tf.reshape(tf.convert_to_tensor(self.y_train_transf['global']), (1, self.T, self.N['global']))
        for idx, var in enumerate(self.input_vars):
          self.input[var] = Input(shape=(None, int(self.N['global'])), name=f'{var}_input')
          self.input_data[var] = tf.reshape(tf.convert_to_tensor(self.x_train_transf[idx]['global']), (1, self.T, self.N['global']))


    self.Mask = tf.reshape(
    tf.convert_to_tensor(self.mask['global']),
                        (1, self.input_data[self.input_vars[0]].shape[1], self.N['global']) )


    self.time_periods = np.arange(1, self.T + 1, 1)

    use_country_trends = bool(getattr(self, "country_trends", False))
    use_quadratic = use_country_trends and bool(getattr(self, "quadratic_trends", True))

    # Country-grouped trends: estimate one trend per country instead of per fid.
    # Only active for genuine 'ee' estimation runs (country_map is set there);
    # for 'wb' data and Monte Carlo it stays None, so the per-fid path is used.
    country_map = getattr(self, "country_map", None)
    group_by_country = (
        use_country_trends
        and bool(getattr(self, "group_trends_by_country", False))
        and str(getattr(self, "data_source", "wb")).lower() == "ee"
        and country_map is not None
    )
    self.trend_groups = None

    if self.holdout == 0:
        dummies_layer = Dummies(
            self.N['global'],
            self.T,
            self.time_periods_na['global'],
            country_trends=use_country_trends
        )
        Delta1, Delta2 = dummies_layer(self.input[self.input_vars[0]])

        country_FE, time_FE, self.country_FE_layer, self.time_FE_layer = create_fixed_effects(self, Delta1, Delta2)

        if use_country_trends:
            trend_layer = CountryTimeTrends(name="country_time_trends")
            linear_trend_input, quadratic_trend_input = trend_layer([Delta1, Delta2])

            grouping_matrix = None
            if group_by_country:
                grouping_matrix, self.trend_groups = build_country_grouping(
                    self.individuals['global'], country_map
                )

            linear_trend, quadratic_trend, self.linear_trend_layer, self.quadratic_trend_layer = create_country_trends(
                self,
                linear_trend_input,
                quadratic_trend_input,
                use_quadratic=use_quadratic,
                grouping_matrix=grouping_matrix,
            )

      # Vectorize the inputs
    for var in self.input_vars:
        self.input_vector[var]= Vectorize(self.N['global'], var)(self.input[var])

    if self.dynamic_model:
      time_input=Vectorize(self.N['global'], 'time', time_periods=self.time_periods)(self.input_vector[self.input_vars[0]])
      input_first= concatenate([[self.input_vector[var] for var in self.input_vars], time_input], axis=2)
    else:
      input_first= concatenate([self.input_vector[var] for var in self.input_vars], axis=2)


    # neural network model
    input_last=create_hidden_layers(self, input_first)

    # Creating temporary output layer, without fixed effects
    output_tmp = create_output_layer(self, input_last)

    if self.holdout==0:
        if self.dynamic_model:
            components = [country_FE, output_tmp]
            if use_country_trends:
              components.append(linear_trend)
              if use_quadratic:
                components.append(quadratic_trend)
            output = Add()(components)
        else:
          components = [time_FE, country_FE, output_tmp]
          if use_country_trends:
            components.append(linear_trend)
            if use_quadratic:
              components.append(quadratic_trend)
          output = Add()(components)
    else:
      output = output_tmp



    # Creating the final output matrix with the correct dimensions
    output_matrix = Matrixize(N=self.N['global'], T=self.T, mask=self.Mask, holdout=self.holdout, n_obs_train=self.noObs["train"])(output)

    # Compiling the model
    self.model = Model(inputs=[self.input[self.input_vars[idx]] for idx, var in enumerate(self.input_vars)], outputs=output_matrix)



    # Counting number of parameters
    self.m = Count_params(self)

    #setting up the visualisation of the model, and the prediction model
    self.model_pred=pred_model(self)
    self.model_visual=Visual_model(self)

