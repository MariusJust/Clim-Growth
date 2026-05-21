from collections import defaultdict
import tensorflow as tf
def initialize_parameters(self):
      
        self.individuals = {}
        self.N = {}
        self.noObs = {}

        self.time_periods_na = {}
        self.time_periods_not_na = {}

        self.in_sample_pred = {}
        self.R2 = {}
        self.MSE = {}
        
        self.alpha = defaultdict(dict)
        self.beta = defaultdict(dict)
 
  
        self.y_train_df = defaultdict(dict)
        self.y_train_transf = defaultdict(dict)
        
        self.y_val_df = defaultdict(dict)
        self.y_val_transf = defaultdict(dict)
        
        self.y_train_val_df = defaultdict(dict)
        self.y_train_val_transf = defaultdict(dict)
        

        self.x_train_transf = defaultdict(dict)
        self.x_val_transf = defaultdict(dict)
        self.x_train_val_transf = defaultdict(dict)
        

        self.mask = defaultdict(dict)
        self.input_data_train = defaultdict(dict)
        self.input_data_val = defaultdict(dict)
        self.input_data = defaultdict(dict)
        
        self.n_vars= len(self.input_vars)
        
        
        self.input = defaultdict(dict)
        
        self.input_vector = defaultdict(dict)

        self.in_sample_loss = None
        self.holdout_loss = None
        self.epochs = None
        self.params = None
        self.BIC = None
        self.AIC = None
        self.P_matrix= None
        

        self.model_pred = None
        # tf.print(self.x_val is not None)
        if self.x_val is not None:
            self.T = self.x_train_val[0]['global'].shape[0]

            self.time_periods = self.x_train_val[0]['global'].index.values
        else:
            self.T = self.x_train[0]['global'].shape[0]

            self.time_periods = self.x_train[0]['global'].index.values
        # tf.print(">>> Initialised T:", self.T)
        self.Depth=len(self.node)
        
        