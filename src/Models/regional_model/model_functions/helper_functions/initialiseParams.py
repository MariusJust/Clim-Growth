from collections import defaultdict
import tensorflow as tf
def initialize_parameters(self):
      
        self.individuals = {}
        self.N = {}
        self.noObs = defaultdict(dict)

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


        self.holdout_loss = None
        self.params = None
        self.BIC = None
        self.AIC = None
        self.country_FE= None
        self.P_matrix= None
        
        self.regions = list(self.x_train[0].keys())
        
        #number of regions
        self.no_regions = len(self.regions)
        if self.x_val is not None:
                self.T = self.x_train_val[0][self.regions[0]].shape[0]
                self.time_periods = self.x_train_val[0][self.regions[0]].index.values
        else:   
                self.T = self.x_train[0][self.regions[0]].shape[0]
                self.time_periods = self.x_train[0][self.regions[0]].index.values
        self.model_pred = None
        
        self.Depth=len(self.node)
        
        