import os
import numpy as np
import tensorflow as tf
import random
from utils.miscelaneous.warnings import turn_off_warnings
from models.helper_functions.regional_model import load_data
from models.helper_functions.shared import fid_country_map
from models import MultivariateModelRegional as Model


turn_off_warnings()

class MainLoop:
    def __init__(self, parent, node):

        self.cfg=parent.cfg
        self.data=parent.data
        self.node= node
        self.run_dir=parent.run_dir
        self.models_tmp = np.zeros(self.cfg.no_inits, dtype=object)
        self.BIC_list = np.zeros(self.cfg.no_inits)
        self.AIC_list = np.zeros(self.cfg.no_inits)
        self.holdout_MSE = np.zeros(self.cfg.no_inits)


        #build a factory for the model, so we don't have to re-initialize the model each time
        self.factory = Model(
            node=None,
            cfg=self.cfg,
            x_train=None,
            y_train=None,
            x_train_val=None,
            y_train_val=None,
            x_val=None,
            y_val=None
        )
        # Load data
        if self.data is not None: #ie we are running a Monte Carlo experiment
            from simulations.simulation_functions import Pivot
            self.growth, self.precip, self.temp = Pivot(self.data)
            # Monte Carlo data is country-level; per-fid country grouping does not apply.
            self.country_map = None
        else:
            self.growth, self.precip, self.temp = load_data('IC', formulation=self.cfg.formulation, data_source=self.cfg.data_source, end_year=self.cfg.data_end)
            self.country_map = fid_country_map() if str(self.cfg.data_source).lower() == 'ee' else None

        self.factory.country_map = self.country_map


    def run_experiment(self):

        self.setup_model_params()

        #loop over initializations
        for j in range(self.cfg.no_inits):

            current_seed = self.cfg.seed_value + j  # update seed
            tf.random.set_seed(current_seed)
            np.random.default_rng(current_seed)
            random.seed(current_seed)


            model_instance=self.factory.get_model()
            model_instance.fit(lr=self.cfg.lr, min_delta=self.cfg.min_delta, patience=self.cfg.patience, verbose=self.cfg.verbose)


            if self.cfg.holdout>0:
                self.models_tmp[j] = model_instance
                self.holdout_MSE[j] = model_instance.holdout_loss
            else:
                model_instance.in_sample_predictions()
                self.models_tmp[j] = model_instance

                #saves the information criteria
                self.BIC_list[j] = model_instance.BIC
                self.AIC_list[j] = model_instance.AIC

            print(f"Initialization {j+1}/{self.cfg.no_inits} for node {self.node} done")

        # Select the best initialization based on BIC (or AIC)
        best_idx_BIC = int(np.argmin(self.BIC_list))
        best_idx_AIC = int(np.argmin(self.AIC_list))
        best_idx_holdout = int(np.argmin(self.holdout_MSE))

        # Create directory if it doesn't exist
        path=f"{self.run_dir}/parameters/{self.node}.weights.h5"
        dir_path = os.path.dirname(path)
        os.makedirs(dir_path, exist_ok=True)

        self.models_tmp[best_idx_BIC].save_params(path)

        if self.cfg.holdout == 0:
            best = self.models_tmp[best_idx_BIC]
            for region in best.regions:
                best.beta[region].to_csv(f"{self.run_dir}/parameters/{self.node}.{region}.Time_FE.csv")
                best.alpha[region].to_csv(f"{self.run_dir}/parameters/{self.node}.{region}.Country_FE.csv")

            if bool(getattr(self.cfg, "country_trends", False)):
                use_quadratic = bool(getattr(self.cfg, "quadratic_trends", True))
                for region in best.regions:
                    best.linear_trend[region].to_csv(f"{self.run_dir}/parameters/{self.node}.{region}.linear_trend.csv")
                    if use_quadratic:
                        best.quadratic_trend[region].to_csv(f"{self.run_dir}/parameters/{self.node}.{region}.quadratic_trend.csv")

        return self.holdout_MSE[best_idx_holdout], self.BIC_list[best_idx_BIC], self.AIC_list[best_idx_AIC], self.node



    def setup_model_params(self):
        if self.cfg.holdout > 0:
            self.factory.x_train = {0: self.temp, 1: self.precip}
            self.factory.y_train = self.growth

            temp_train_val = {key: df.iloc[:-self.cfg.holdout, :] for key, df in self.temp.items()}
            temp_val = {key: df.iloc[-self.cfg.holdout:, :] for key, df in self.temp.items()}
            precip_train_val = {key: df.iloc[:-self.cfg.holdout, :] for key, df in self.precip.items()}
            precip_val = {key: df.iloc[-self.cfg.holdout:, :] for key, df in self.precip.items()}
            growth_train_val = {key: df.iloc[:-self.cfg.holdout, :] for key, df in self.growth.items()}
            growth_val = {key: df.iloc[-self.cfg.holdout:, :] for key, df in self.growth.items()}

            self.factory.x_train_val = {0: temp_train_val, 1: precip_train_val}
            self.factory.y_train_val = growth_train_val
            self.factory.x_val = {0: temp_val, 1: precip_val}
            self.factory.y_val = growth_val

        else:
            self.factory.x_train = {0: self.temp, 1: self.precip}
            self.factory.y_train = self.growth

        self.factory.node = self.node
