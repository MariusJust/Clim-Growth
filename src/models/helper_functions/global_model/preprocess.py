import numpy as np

def Preprocess(self):     
    compute_x_train_fullSample(self, self.x_train)      
    compute_y_train_fullSample(self, self.y_train)
    compute_mask_fullSample(self, self.x_train)
    
    if self.x_val is not None and self.y_val is not None:
        compute_x_train_val(self)      
        compute_y_train_val(self)
        compute_mask_train_val(self, self.x_train_val)

################################### Full Sample Preprocessing Functions ###################################
def compute_x_train_fullSample(self, x_train):

        self.individuals['global'] = x_train[0]['global'].columns.values
        self.N['global'] = len(self.individuals['global'])   
                
        for key, var in enumerate(x_train):
            self.x_train_transf[key]['global'] =np.array(x_train[key]['global'].copy())
            

def compute_y_train_fullSample(self, y_train):
    self.y_train_df['global'] = y_train['global'].copy()
    self.y_train_transf['global'] = np.array(y_train['global'].copy())
    
        
def compute_mask_fullSample(self, x_train):

    self.mask['global'] = np.isnan(x_train[0]['global'])
    
    # also calculate the number of non-NaN values
    non_na = np.sum(~np.isnan(x_train[0]['global']), axis=1) > 0
    self.time_periods_not_na['global'] = non_na
    self.time_periods_na['global'] = np.sum(~non_na)
    self.noObs['global'] = self.N['global']* len(self.x_train_transf[0]['global']) - np.isnan(self.x_train_transf[0]['global']).sum()
    
################################### Train/val Preprocessing Functions ###################################

 
def compute_x_train_val(self):
     
  # compute the training set for the validation
        for key, var in enumerate(self.x_train_val):

            self.x_train_val_transf[key]['global'] = np.array(self.x_train_val[key]['global'].copy())
            self.x_val_transf[key]['global'] = np.array(self.x_val[key]['global'].copy())
       
            

def compute_y_train_val(self):

        self.y_train_val_df['global'] = self.y_train_val['global'].copy()
        self.y_train_val_transf['global'] = np.array(self.y_train_val_df['global'].copy())
            
        self.y_val_df['global'] = self.y_val['global'].copy()
        self.y_val_transf['global'] = np.array(self.y_val['global'].copy())
        
   


def compute_mask_train_val(self, x_train):

    self.mask['train'] = np.isnan(x_train[0]['global'])
    
    # also calculate the number of non-NaN values
    non_na = np.sum(~np.isnan(x_train[0]['global']), axis=1) > 0
    self.time_periods_not_na['train'] = non_na
    self.time_periods_na['train'] = np.sum(~non_na)
    self.noObs['train'] = self.N['global']* self.T - np.isnan(self.x_train_val_transf[0]['global']).sum()





