import multiprocessing as mp
from pyexpat import model
from .builders import build_arg_list_mc
from .manager import setup_manager
import time
from utils import create_pred_input
import numpy as np
from types import SimpleNamespace

class MultiprocessingMC:
    """
    Class to handle multiprocessing for Monte Carlo simulations.
    """
    
    def __init__(self, cfg, specification, model, run_dir: str, node_index=None, nodes_list=None):
        self.node_index = node_index
        self.nodes_list = nodes_list
        self.cfg = cfg 
        self.specification = specification
        self.model = model
        self.run_dir = run_dir # Initialize run_dir to None, will be set in run() method
    
    def run(self):
        
        # record start time for runtime tracking
        self.start_time = time.time()

        #manager for shared data dictionary
        setup_manager(self)
        
        #builds arguments list to pass to the mc_worker
        build_arg_list_mc(self)
        

        self.counter= self.manager.Value('i', 0)

        #define the number of processes to use
        pool = mp.Pool(processes=self.cfg.instance.n_process)
        
        self.all_surfaces = []
        self.all_country_FE = []
        self.all_time_FE = []
        self.all_linear_trends = []
        self.all_quadratic_trends = []
        
        def error_cb(exc):
            import traceback, sys
            print("WORKER ERROR (in child):", exc, file=sys.stderr)
            traceback.print_exc()

        def callback_one(result):
            try:
                best_surface, country_FE, time_FE, linear_trend, quadratic_trend, runtime = result
                # ensure values exist
                try:
                    with self.lock:
                        self.counter.value += 1
                        self.all_surfaces.append(best_surface)
                        self.all_country_FE.append(country_FE)
                        self.all_time_FE.append(time_FE)
                        self.all_linear_trends.append(linear_trend)
                        self.all_quadratic_trends.append(quadratic_trend)
                except AttributeError:
                    # fallback if lock is not a context manager
                    self.lock.acquire()
                    try:
                        self.counter.value += 1
                        self.all_surfaces.append(best_surface)
                        self.all_country_FE.append(country_FE)
                        self.all_time_FE.append(time_FE)
                        self.all_linear_trends.append(linear_trend)
                        self.all_quadratic_trends.append(quadratic_trend)
                    finally:
                        self.lock.release()

                runtime = runtime/60
                elapsed_time = (time.time() - self.start_time)/60
                print(f"Process {self.counter.value}/{self.cfg.mc.reps} completed in {runtime: .2f} min. Total elapsed time: {elapsed_time: .2f} min.", flush=True)

            except Exception as e:
                import traceback
                print("Error in callback_one:", e)
                traceback.print_exc()

        # then in loop:
        for args_dict in self.rep_args:
            pool.apply_async(
                mc_worker,
                kwds=args_dict,
                callback=callback_one,
                error_callback=error_cb
            )


        pool.close()
        pool.join()

        return self.all_surfaces, self.all_country_FE, self.all_time_FE, self.all_linear_trends, self.all_quadratic_trends


def mc_worker(**payload):
    import time
    import numpy as np
    from utils import create_pred_input

    class ConfigShim(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as e:
                raise AttributeError(name) from e

        def __setattr__(self, name, value):
            self[name] = value

    class ParentShim:
        def __init__(self, cfg, data=None, run_dir=None):
            self.cfg = cfg
            self.data = data
            self.run_dir = run_dir

    t0 = time.time()

    data = payload.get("data")
    node = payload.get("node")
    run_dir = payload.get("run_dir")
    model = payload.get("model")
    model_selection = payload.get("model_selection")

    mean_T = np.mean(data["temperature"])
    std_T = np.std(data["temperature"])
    mean_P = np.mean(data["precipitation"])
    std_P = np.std(data["precipitation"])

 
    time_periods = data["Year"].max() - data["Year"].min() + 1

    if payload.get("dynamic_model"):
        pred_input, T, P = create_pred_input(
            mc=True,
            mean_T=mean_T,
            std_T=std_T,
            mean_P=mean_P,
            std_P=std_P,
            time_periods=time_periods
        )
    else:
        pred_input, T, P = create_pred_input(
            mc=True,
            mean_T=mean_T,
            std_T=std_T,
            mean_P=mean_P,
            std_P=std_P,
            time_periods=None
        )

    if model == "NN":
        from models.global_model.run_experiment_ic import MainLoop
        allowed_cfg = (
            "no_inits", "seed_value", "lr", "min_delta", "patience",
            "verbose", "dropout", "n_splits", "cv_approach",  "dynamic_model",
            "model_selection", "formulation", "data_source", "holdout", "input_vars",
            "activation", "country_trends"
        )

        cfg_dict = {k: payload[k] for k in allowed_cfg if k in payload}
        cfg_obj = ConfigShim(**cfg_dict)
        parent = ParentShim(cfg=cfg_obj, data=data, run_dir=run_dir)

       
        main_loop = MainLoop(parent, node, data=data)

        *unused, best_surface, country_FE, time_FE, linear_trend, quadratic_trend = main_loop.run_experiment()

        runtime = time.time() - t0
        preds = best_surface.predict({"X_in": pred_input}, verbose=0).reshape(-1,)
        return preds, country_FE, time_FE, linear_trend, quadratic_trend, runtime

     

    elif model == "Quadratic":
        from simulations.simulation_functions import quadratic_model
        predictions = quadratic_model(data, P, T)
        return predictions, None, None, None, None, time.time() - t0

    else:
        from simulations.simulation_functions import interaction_model
        predictions = interaction_model(data, P, T)
        return predictions, None, None, None, None, time.time() - t0
