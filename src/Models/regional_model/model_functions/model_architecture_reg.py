from tensorflow.keras.layers import Input, Add, concatenate
from tensorflow.keras import Model
import tensorflow as tf
from models.regional_model.model_functions.helper_functions import Dummies, create_fixed_effects, Vectorize, Count_params, Matrixize, create_hidden_layers, create_output_layer, Visual_model, individual_loss


class Regions:
    def __init__(self, parent, regions):
       self.regions=regions
       self.parent=parent
      
      #unpack parent attributes
       for key, value in vars(parent).items():
            setattr(self, key, value)
            
       self.region_builders = []

      
    def SetupRegionalModel(self):
    
        for region in self.regions:
          region_model=BuildRegion(parent=self, region=region)
          self.region_builders.append(region_model)

        #extract model lists: 
        self.extract_lists()
        
        
        self.parent.model = Model(inputs=[self.parent.inputs_temp, self.parent.inputs_precip], outputs=self.parent.output_matrix)  

        self.parent.m=Count_params(self.parent)
        
        # compute the visual model for each region
        self.parent.model_visual={}
        for region_model in self.region_builders:
            self.parent.model_visual[region_model.region]=Visual_model(self.parent, region_model)
       
    def extract_lists(self):
        self.parent.inputs_temp =   [b.input_temp for b in self.region_builders]
        self.parent.inputs_precip = [b.input_precip for b in self.region_builders]
        self.parent.output_matrix = [b.output_matrix for b in self.region_builders]
        self.parent.input_data_temp = [b.input_data_temp for b in self.region_builders]
        self.parent.input_data_precip = [b.input_data_precip for b in self.region_builders]
        self.parent.targets = [b.target for b in self.region_builders]
        self.parent.loss_list = [b.loss_fn for b in self.region_builders]
        self.parent.masks = [b.Mask for b in self.region_builders]
        if self.holdout>0:
            self.parent.input_data_temp_train = [b.input_data_temp_train for b in self.region_builders]
            self.parent.input_data_precip_train = [b.input_data_precip_train for b in self.region_builders]
            self.parent.targets_train = [b.targets_train for b in self.region_builders]
            self.parent.input_data_temp_val = [b.input_data_temp_val for b in self.region_builders]
            self.parent.input_data_precip_val = [b.input_data_precip_val for b in self.region_builders]
            self.parent.targets_val = [b.targets_val for b in self.region_builders]
        else:
            self.parent.country_FE_layer=[b.country_FE_layer for b in self.region_builders]
            self.parent.time_FE_layer=[b.time_FE_layer for b in self.region_builders]
        
        
    
class BuildRegion:
    def __init__(self, parent, region):
      
      self.region=region
      
      #unpack parent attributes
      for key, value in vars(parent).items():
            setattr(self, key, value)
      
      self.input_temp = Input(shape=(None, self.N[region]), name=f"temp_in_{region}")
      self.input_precip = Input(shape=(None, self.N[region]), name=f"precip_in_{region}")
      
      
      if self.x_val is not None:
        self.input_data_temp_train = tf.reshape(tf.convert_to_tensor(self.x_train_val[0][region]), (1, self.T, self.N[region]))
        self.input_data_precip_train = tf.reshape(tf.convert_to_tensor(self.x_train_val[1][region]), (1, self.T, self.N[region]))
        self.targets_train = tf.reshape(tf.convert_to_tensor(self.y_train_val[region]), (1, self.T, self.N[region]))
        
        self.input_data_temp_val = tf.reshape(tf.convert_to_tensor(self.x_val[0][region]), (1, self.holdout, self.N[region]))
        self.input_data_precip_val = tf.reshape(tf.convert_to_tensor(self.x_val[1][region]), (1, self.holdout, self.N[region]))
        self.targets_val = tf.reshape(tf.convert_to_tensor(self.y_val[region]), (1, self.holdout, self.N[region]))
        
        self.input_data_temp = tf.reshape(tf.convert_to_tensor(self.x_train[0][region]), (1, self.T+self.holdout, self.N[region]))
        self.input_data_precip = tf.reshape(tf.convert_to_tensor(self.x_train[1][region]), (1, self.T+self.holdout, self.N[region]))    
        self.target = tf.reshape(tf.convert_to_tensor(self.y_train[region]), (1, self.T+self.holdout, self.N[region]))
        
        
      else:
        
        self.input_data_temp = tf.reshape(tf.convert_to_tensor(self.x_train_transf[0][region]), (1, self.T, self.N[region]))
        self.input_data_precip = tf.reshape(tf.convert_to_tensor(self.x_train_transf[1][region]), (1, self.T, self.N[region]))    
        self.target = tf.reshape(tf.convert_to_tensor(self.y_train_transf[region]), (1, self.T, self.N[region])) 
        
    
      self.Mask = tf.reshape(
      tf.convert_to_tensor(self.mask[region]),
                          (1, self.input_data_temp.shape[1], self.N[region]) )
      
      self.loss_fn = individual_loss(self.Mask)
      
      if self.holdout==0:
        dummies_layer = Dummies(self.N[region], self.T, self.time_periods_na[region], country_trends=False)
        Delta1, Delta2 = dummies_layer(self.input_temp)

        # Creating fixed effects
        country_FE, time_FE, self.country_FE_layer, self.time_FE_layer = create_fixed_effects(self, Delta1, Delta2)
          
        # Vectorize the inputs
      temp_input=Vectorize(self.N[region], 'temp')(self.input_temp)
      precip_input= Vectorize(self.N[region], 'precip')(self.input_precip)

      if self.dynamic_model:
        time_input=Vectorize(self.N[region], 'time', time_periods=self.time_periods)(self.input_temp)
        input_first= concatenate([temp_input, precip_input, time_input], axis=2)
      else:
        input_first= concatenate([temp_input, precip_input], axis=2)
    
      # neural network model
      input_last=create_hidden_layers(self, input_first)
  
      # Creating temporary output layer, without fixed effects
      output_tmp = create_output_layer(self, input_last)
      
      if self.holdout==0:
        if self.dynamic_model:
          output = Add()([country_FE, output_tmp])
        else:
          output = Add()([time_FE, country_FE, output_tmp])
      else:
        output = output_tmp
      # Creating the final output matrix with the correct dimensions
      self.output_matrix = Matrixize(N=self.N[region], T=self.T, mask=self.Mask, holdout=self.holdout, n_obs_train=self.noObs["train"][region] )(output)

        
        
        