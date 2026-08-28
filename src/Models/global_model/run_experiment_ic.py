import os
import numpy as np
import tensorflow as tf
import random
from pathlib import Path
import pandas as pd
from utils.miscelaneous.warnings import turn_off_warnings
from models.helper_functions.global_model.diagnostics import save_init_diagnostics, save_summary_diagnostics
from models.helper_functions.global_model import load_data
from models.helper_functions.shared import fid_country_map
from models import MultivariateModelGlobal as Model
import time


turn_off_warnings()

class MainLoop:
    def __init__(self, parent, node, data=None):
   
        self.cfg=parent.cfg
        self.data=parent.data
        self.run_dir = parent.run_dir
        self.node= node
        self.models_tmp = np.zeros(self.cfg.no_inits, dtype=object)
        self.BIC_list = np.zeros(self.cfg.no_inits)
        self.AIC_list = np.zeros(self.cfg.no_inits)
      
        
        #build a factory for the model, so we don't have to re-initialize the model each time
        self.factory = Model(
            node=None, 
            cfg=self.cfg,
            x_train=None,     
            y_train=None,
            x_train_val=None,
            y_train_val=None,
            x_val=None,
            y_val=None,
     
        )

        
        # Load data
        if self.data is not None: #ie we are running a Monte Carlo experiment
            from simulations.simulation_functions import Pivot
            self.growth, self.precip, self.temp = Pivot(self.data)
            # Monte Carlo data is country-level; per-fid country grouping does not apply.
            self.country_map = None
        else:
            self.growth, self.precip, self.temp = load_data('IC', self.cfg.data_source, end_year=self.cfg.data_end, target_mode=getattr(self.cfg, 'target_mode', 'growth'))
            self.country_map = fid_country_map() if str(self.cfg.data_source).lower() == 'ee' else None

        self.factory.country_map = self.country_map

    def _build_model_instance(self):
        model_factory = Model(
            node=self.node,
            cfg=self.cfg,
            x_train=self.factory.x_train,
            y_train=self.factory.y_train,
            x_train_val=self.factory.x_train_val,
            y_train_val=self.factory.y_train_val,
            x_val=self.factory.x_val,
            y_val=self.factory.y_val,
        )
        model_factory.country_map = self.country_map
        return model_factory.get_model()
   
   
    def run_experiment(self):   
        self.factory.x_train = {0: self.temp, 1: self.precip}
        self.factory.y_train = self.growth
            
        self.factory.node = self.node

        use_diagnostics = self.data is None and bool(getattr(self.cfg, "use_diagnostics", False))
        diagnostics_dir = Path(self.run_dir) / "diagnostics" / str(self.node)
        if use_diagnostics:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

        init_rows = []
        
        #loop over initializations
        for j in range(self.cfg.no_inits):
            time_start = time.time()
        
            current_seed = self.cfg.seed_value + j  # update seed
            tf.random.set_seed(current_seed)
            np.random.default_rng(current_seed)
            random.seed(current_seed)

            
            if self.data is None:
                model_instance = self._build_model_instance()
            else:
                model_instance = self.factory.get_model()

            model_instance.fit(lr=self.cfg.lr, min_delta=self.cfg.min_delta, patience=self.cfg.patience, verbose=self.cfg.verbose)
            history_frame = pd.DataFrame(model_instance.model.history.history)
         
            model_instance.in_sample_predictions()
         
            self.models_tmp[j] = model_instance

            #saves the information criteria
            self.BIC_list[j] = model_instance.BIC
            self.AIC_list[j] = model_instance.AIC

            if use_diagnostics:
                if history_frame is not None:
                    init_rows.append(
                        save_init_diagnostics(
                            model_instance=model_instance,
                            history_frame=history_frame,
                            diagnostics_dir=diagnostics_dir,
                            init_index=j,
                            seed_value=current_seed,
                            bic=self.BIC_list[j],
                            aic=self.AIC_list[j],
                            R2=model_instance.R2['global'] if model_instance.R2 is not None and 'global' in model_instance.R2 else None,
                        )
                    )
            
            time_end = time.time()  
              
            
            elapsed = int(time_end - time_start)

            hours, remainder = divmod(elapsed, 3600)
            minutes, seconds = divmod(remainder, 60)

            print(f"Finished node {self.node} with initialization {j+1}/{self.cfg.no_inits} in {hours} hours, {minutes} minutes, and {seconds} seconds")      
                    
        # Select the best initialization based on BIC (or AIC)
        
        best_idx_BIC = int(np.argmin(self.BIC_list))
        best_idx_AIC = int(np.argmin(self.AIC_list))
        
    
        #only save the model parameters if the data is the real data, and not simulated data
        if self.data is None:

            print(f"saving model parameters to: {self.run_dir}/parameters/{self.node}.weights.h5")

            # Create directory if it doesn't exist
            path=f"{self.run_dir}/parameters/{self.node}.weights.h5"
            dir_path = os.path.dirname(path)

            os.makedirs(dir_path, exist_ok=True)

            best_idx_save = best_idx_BIC

            best_model_for_outputs = self.models_tmp[best_idx_save]
            if use_diagnostics:
                best_snapshot_path = diagnostics_dir / f"init_{best_idx_save}.weights.h5"
                best_model = self._build_model_instance()
                best_model.load_params(str(best_snapshot_path))
                best_model.in_sample_predictions()
                best_model.save_params(path)
            else:
                best_model = best_model_for_outputs
                best_model.save_params(path)

            #also save the time, country fixed effects and the country trends
            # (the dynamic model has no additive time FE: beta is None, skip it)
            if best_model.beta is not None:
                best_model.beta.to_csv(f"{self.run_dir}/parameters/{self.node}.Time_FE.csv")
            best_model.alpha.to_csv(f"{self.run_dir}/parameters/{self.node}.Country_FE.csv")
            if bool(getattr(self.cfg, "country_trends", False)):
                use_quadratic = bool(getattr(self.cfg, "quadratic_trends", True))
                best_model.linear_trend.to_csv(f"{self.run_dir}/parameters/{self.node}.linear_trend.csv")
                if use_quadratic:
                    best_model.quadratic_trend.to_csv(f"{self.run_dir}/parameters/{self.node}.quadratic_trend.csv")

            if use_diagnostics:
                save_summary_diagnostics(
                    diagnostics_dir=diagnostics_dir,
                    init_rows=init_rows,
                    best_idx=best_idx_save,
                    recorded_aic=self.AIC_list[best_idx_save],
                    recomputed_aic=best_model.AIC,
                )
            return np.nan, self.BIC_list[best_idx_BIC], self.AIC_list[best_idx_AIC], self.node
        else: #Monte carlo simulation
            best_surface=self.models_tmp[best_idx_BIC].model_visual
            country_FE = self.models_tmp[best_idx_BIC].alpha_dict
            time_FE = self.models_tmp[best_idx_BIC].beta_dict
            linear_trend = None
            quadratic_trend = None
            if bool(getattr(self.cfg, "country_trends", False)):
                linear_trend = self.models_tmp[best_idx_BIC].linear_trend_dict
                if bool(getattr(self.cfg, "quadratic_trends", True)):
                    quadratic_trend = self.models_tmp[best_idx_BIC].quadratic_trend_dict
            return np.nan, self.BIC_list[best_idx_BIC], self.AIC_list[best_idx_AIC], self.node, best_surface, country_FE, time_FE, linear_trend, quadratic_trend

