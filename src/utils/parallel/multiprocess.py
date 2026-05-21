from multiprocessing import Pool
from tqdm import tqdm
from multiprocessing import TimeoutError
from .builders import build_arg_list_ic
import ast


class Multiprocess:

    """
    This class runs information criteria (IC) based model selection in parallel.
    It initializes with configuration parameters (from the config folder) and data, builds the argument list for each node, and executes the training in parallel.
    The results are stored in a dictionary where keys are node indices and values are lists containing either BIC/AIC values or holdout errors.
    """
    def __init__(self, cfg, run_dir, data=None):
        self.Model_selection = cfg.model_selection
        self.nodes_list = [ast.literal_eval(s) for s in cfg.nodes_list]
        self.cfg=cfg
        self.data = data
        self.run_dir = run_dir

    def run(self):

        if self.Model_selection == 'IC' or self.Model_selection == 'Holdout':
            build_arg_list_ic(self)
        else:
            raise ValueError("Model_selection must be either 'IC' or 'Holdout'")


        print(f"Starting parallel processing with {self.cfg.n_process} processes...")
        results= self.parallel_execution()

        return results


    def parallel_execution(self):

            self.storage = {}

            pool = Pool(self.cfg.n_process)
            async_results = [
                pool.apply_async(self.worker, kwds={'node': self.nodes_list[i], 'data': self.data})
                for i in range(len(self.nodes_list))
            ]
            pool.close()

            for i, async_result in enumerate(tqdm(async_results, desc="Processing nodes", unit="node")):
                try:
                    result = async_result.get(timeout=self.cfg.timeout_per_node)
                except TimeoutError:
                    print(f"Timeout occurred for node {i}")
                    self.storage[i]=None
                    continue

                if self.Model_selection == 'Holdout':
                    holdout_error, node = result
                    self.storage[node] = [holdout_error]
                else:
                    bic, aic, node = result
                    self.storage[node] = [bic,aic]

            pool.terminate()
            pool.join()

            return self.storage


    def worker(self, node, data=None):
        if self.cfg.formulation == 'regional' or self.cfg.formulation == 'income':
            from models.regional_model.run_experiment_ic import MainLoop as MainLoop
            model_loop = MainLoop(self, node)
            if self.Model_selection == 'Holdout':
                Holdout_error, BIC, AIC, node = model_loop.run_experiment()
                return Holdout_error, node
            else:
                Holdout_error, BIC, AIC, node= model_loop.run_experiment()
                return BIC, AIC, node
        else:
            from models.global_model.run_experiment_ic import MainLoop as MainLoop

            if data is not None:
                model_loop = MainLoop(self, node, data=data)
                # Monte Carlo experiment
                Holdout_error, BIC, AIC, node, *_ = model_loop.run_experiment()
                return BIC, AIC, node
            else:
                model_loop = MainLoop(self, node)
                if self.Model_selection == 'Holdout':
                    Holdout_error,_,_, node= model_loop.run_experiment()
                    return Holdout_error, node
                else:
                    _, BIC, AIC, node = model_loop.run_experiment()
                    return BIC, AIC, node
