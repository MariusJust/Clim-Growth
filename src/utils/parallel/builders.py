import os
import numpy as np


 
def build_arg_list_cv(self):
        self.arg_list=[(
        self.nodes_list[i], 
        self.cfg.no_inits, 
        self.cfg.seed_value, 
        self.cfg.lr,
        self.cfg.min_delta, 
        self.cfg.patience, 
        self.cfg.verbose, 
        self.cfg.dropout,
        self.cfg.n_splits, 
        self.cfg.cv_approach, 
        self.cfg.n_countries, 
        self.cfg.time_periods,
        self.cfg.dynamic_model,
        self.data_source,
          
        self.data
    ) for i in range(len(self.nodes_list))]


def build_arg_list_ic(self):
        self.arg_list=[(
        self.nodes_list[i],
        self.cfg.no_inits, 
        self.cfg.seed_value, 
        self.cfg.lr,
        self.cfg.min_delta, 
        self.cfg.patience, 
        self.cfg.verbose, 
        self.cfg.dropout,
        self.cfg.dynamic_model,
        self.cfg.data_source,
        #we have the data end year in the config file, so we can use that to filter out the data after that year.   
        self.data[self.data['Year']<=self.cfg.data_end]
        ) for i in range(len(self.nodes_list))]
            
  
def build_arg_list_mc(self):
    from simulations.simulation_functions import simulate

    fixed_effects = None
    if self.cfg.mc.sample_data:
        country_path = os.path.join(self.run_dir, "country_effect_absolute.npy")
        time_path = os.path.join(self.run_dir, "time_trend_absolute.npy")
        if os.path.exists(country_path) and os.path.exists(time_path):
            fixed_effects = {
                "country": np.load(country_path, allow_pickle=True).item(),
                "time": np.load(time_path, allow_pickle=True).item(),
            }
    if self.model == "NN":
        self.rep_args = [
            {
                "model": self.model,                               # keep for branching
                "node": self.nodes_list[self.node_index],
                "no_inits": self.cfg.instance.no_inits,
                "seed_value": self.cfg.instance.seed_value + rep + 1,  
                "lr": self.cfg.instance.lr,
                "min_delta": self.cfg.instance.min_delta,
                "patience": self.cfg.instance.patience,
                "verbose": self.cfg.instance.verbose,
                "dropout": self.cfg.instance.dropout,
                "model_selection":self.cfg.instance.model_selection,
                "dynamic_model":self.cfg.instance.dynamic_model,
                "holdout":self.cfg.instance.holdout,
                "input_vars":self.cfg.instance.input_vars,
                "activation":self.cfg.instance.activation,
                "data": simulate(
                    seed=self.cfg.instance.seed_value + rep + 1,
                    specification=self.specification,
                    add_noise=True,
                    sample_data=self.cfg.mc.sample_data,
                    dynamic=self.cfg.instance.dynamic_model,
                    run_dir=self.run_dir,
                    save_effects=self.cfg.mc.sample_data,
                    rep_id=rep,
                    fixed_effects=fixed_effects
                ),
                "run_dir": self.run_dir
            }
            for rep in range(self.cfg.mc.reps)
        ]
    else:
        self.rep_args = [
        {
            "model": self.model,
            "data": simulate(
                seed=self.cfg.instance.seed_value + rep + 1,
                specification=self.specification,
                add_noise=True,
                sample_data=self.cfg.mc.sample_data,
                dynamic=self.cfg.instance.dynamic_model,
                run_dir=self.run_dir,
                save_effects=self.cfg.mc.sample_data,
                rep_id=rep,
                fixed_effects=fixed_effects
            ),
            "run_dir": self.run_dir
        }
        for rep in range(self.cfg.mc.reps)
    ]
    