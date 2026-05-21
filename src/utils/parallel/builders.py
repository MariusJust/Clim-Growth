import os
import numpy as np


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
        self.data
        ) for i in range(len(self.nodes_list))]


def build_arg_list_mc(self):
    from simulations.simulation_functions import simulate

    def load_fixed_effects():
        true_param_dir = os.path.join(self.run_dir, "true_parameters")
        country_path = os.path.join(true_param_dir, "country_effect_absolute.npy")
        time_path = os.path.join(true_param_dir, "time_trend_absolute.npy")
        if not os.path.exists(country_path) or not os.path.exists(time_path):
            country_path = os.path.join(self.run_dir, "country_effect_absolute.npy")
            time_path = os.path.join(self.run_dir, "time_trend_absolute.npy")
        if os.path.exists(country_path) and os.path.exists(time_path):
            loaded_effects = {
                "country": np.load(country_path, allow_pickle=True).item(),
                "time": np.load(time_path, allow_pickle=True).item(),
            }
            linear_path = os.path.join(true_param_dir, "linear_trend_true.npy")
            quadratic_path = os.path.join(true_param_dir, "quadratic_trend_true.npy")
            if not os.path.exists(linear_path) or not os.path.exists(quadratic_path):
                linear_path = os.path.join(self.run_dir, "linear_trend_true.npy")
                quadratic_path = os.path.join(self.run_dir, "quadratic_trend_true.npy")
            if os.path.exists(linear_path) and os.path.exists(quadratic_path):
                loaded_effects["linear_trend"] = np.load(linear_path, allow_pickle=True).item()
                loaded_effects["quadratic_trend"] = np.load(quadratic_path, allow_pickle=True).item()
            return loaded_effects
        return None

    fixed_effects = None
    if self.cfg.mc.sample_data:
        fixed_effects = load_fixed_effects()
        if fixed_effects is None:
            simulate(
                seed=self.cfg.instance.seed_value,
                specification=self.specification,
                add_noise=True,
                sample_data=self.cfg.mc.sample_data,
                dynamic=self.cfg.instance.dynamic_model,
                run_dir=self.run_dir,
                save_effects=True,
                country_trends=self.cfg.instance.country_trends
            )
            fixed_effects = load_fixed_effects()
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
                "country_trends":self.cfg.instance.country_trends,
                "data_source":self.cfg.instance.data_source,
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
                    fixed_effects=fixed_effects,
                    country_trends=self.cfg.instance.country_trends
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
                fixed_effects=fixed_effects,
                country_trends=self.cfg.instance.country_trends
            ),
            "run_dir": self.run_dir
        }
        for rep in range(self.cfg.mc.reps)
    ]
