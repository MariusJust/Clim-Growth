import numpy as np
import pandas as pd


def Preprocess(self):

    compute_x_train_fullSample(self, self.x_train)
    compute_y_train_fullSample(self)
    compute_mask_fullSample(self)

    if self.x_val is not None and self.y_val is not None:
        compute_x_train_val(self)
        compute_y_train_val(self)
        compute_mask_train_val(self, self.x_train_val)


def compute_x_train_fullSample(self, x_train):

     for key, var in x_train.items():
            for region in self.regions:
                x_df_var = self.x_train[key][region]
                x_np_var = x_df_var.values
                self.x_train_transf[key][region] = np.array(x_df_var.copy())

                if key == 0:
                    self.individuals[region] = x_df_var.columns.values
                    self.N[region] =len(self.individuals[region])

                    self.time_periods_not_na[region] = np.sum(~np.isnan(x_np_var), axis=1) > 0
                    self.time_periods_na[region] = np.sum(~self.time_periods_not_na[region])

                    self.noObs[region] = self.N[region]*(self.T+self.holdout) - np.isnan(x_np_var).sum()
                    self.noObs["train"][region] = self.noObs[region]

def compute_y_train_fullSample(self):
        for region in self.regions:
                y_df = self.y_train[region]

                self.y_train_transf[region] = np.array(y_df.copy())
                self.y_train_df[region]=y_df.copy()


def compute_mask_fullSample(self):

     for region in self.regions:
            self.mask[region] = np.isnan(self.x_train[0][region])

            non_na = np.sum(~self.mask[region], axis=1) > 0
            self.time_periods_not_na[region] = non_na
            self.time_periods_na[region] = np.sum(~non_na)

def compute_x_train_val(self):
        for key, var in self.x_train_val.items():
            for region in self.regions:
                self.x_train_val_transf[key][region] = np.array(self.x_train_val[key][region].copy())
                self.x_val_transf[key][region] = np.array(self.x_val[key][region].copy())

def compute_y_train_val(self):
    for region in self.regions:
        self.y_train_val_df[region] = self.y_train_val[region].copy()
        self.y_train_val_transf[region] = np.array(self.y_train_val_df[region].copy())

        self.y_val_df[region] = self.y_val[region].copy()
        self.y_val_transf[region] = np.array(self.y_val_df[region].copy())

def compute_mask_train_val(self, x_train):
    for region in self.regions:
        self.mask['train'][region] = np.isnan(self.x_train_val[0][region])

        self.time_periods_not_na[region] = np.sum(~np.isnan(self.x_train_val[0][region]), axis=1) > 0
        self.time_periods_na[region] = np.sum(~self.time_periods_not_na[region])
        self.noObs["train"][region] = self.N[region]*self.T - np.isnan(self.x_train_val_transf[0][region]).sum()
