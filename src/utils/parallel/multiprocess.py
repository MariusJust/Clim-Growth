from multiprocessing import Pool
from tqdm import tqdm
from multiprocessing import TimeoutError
from .builders import build_arg_list_ic
from collections import defaultdict
import ast


class Multiprocess:

    """
    This class runs information criteria (IC) based model selection in parallel.
    It initializes with configuration parameters (from the config folder) and data, builds the argument list for each node, and executes the training in parallel.
    The results are stored in a dictionary where keys are node indices and values are lists containing either BIC/AIC values or holdout errors.

    Two granularities of parallelism are available.

      * per node (default). One task per architecture; the ``no_inits``
        initialisations run serially inside that task. Fine when the number of
        architectures is comparable to ``n_process``.
      * per (node, init), enabled by ``instance.optim.parallel_over_inits``.
        One task per initialisation. Necessary whenever the sweep has far fewer
        architectures than processes: with 6 architectures on 55 processes the
        per-node scheme leaves 49 cores idle and serialises every initialisation,
        which turns roughly 16 hours of work into an 80 hour job.

    The per-(node, init) path requires ``use_diagnostics: true``. Fitted models
    cannot cross a process boundary, so the winning initialisation is recovered
    by reloading ``diagnostics/<node>/init_<j>.weights.h5``, which only the
    diagnostics writer produces.
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


        if self._init_parallel():
            print(f"Starting parallel processing with {self.cfg.n_process} processes "
                  f"over {len(self.nodes_list)} x {self.cfg.no_inits} (node, init) tasks...")
            return self.parallel_execution_by_init()

        print(f"Starting parallel processing with {self.cfg.n_process} processes...")
        results= self.parallel_execution()

        return results

    def _init_parallel(self):
        """Whether to distribute over (node, init) pairs rather than over nodes."""
        if not bool(getattr(self.cfg, "parallel_over_inits", False)):
            return False
        if self.Model_selection != 'IC':
            return False                      # Holdout selects on a different quantity
        if self.data is not None:
            return False                      # Monte Carlo returns in-memory surfaces
        if str(self.cfg.formulation).lower() != 'global':
            return False                      # regional/income have their own MainLoop
        if not bool(getattr(self.cfg, "use_diagnostics", False)):
            raise ValueError(
                "parallel_over_inits requires use_diagnostics: true, because the "
                "winning initialisation is reloaded from its diagnostics snapshot.")
        return True

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

    def parallel_execution_by_init(self):
        """Fit every (node, init) pair in parallel, then reduce to one winner per node."""

        self.storage = {}

        # ---- pass 1: one task per (node, init) --------------------------------
        tasks = [(node, j)
                 for node in self.nodes_list
                 for j in range(int(self.cfg.no_inits))]

        pool = Pool(self.cfg.n_process)
        async_results = [
            pool.apply_async(self.worker_fit_init, kwds={'node': node, 'init': j})
            for node, j in tasks
        ]
        pool.close()

        rows_by_node = defaultdict(list)
        for (node, j), async_result in zip(tasks, tqdm(async_results, desc="Fitting (node, init)", unit="fit")):
            try:
                _bic, _aic, row = async_result.get(timeout=self.cfg.timeout_per_node)
            except TimeoutError:
                print(f"Timeout occurred for node {node} init {j}")
                continue
            except Exception as exc:                                   # noqa: BLE001
                print(f"Failed node {node} init {j}: {type(exc).__name__}: {exc}")
                continue
            if row is not None:
                rows_by_node[node].append(row)

        pool.terminate()
        pool.join()

        # ---- pass 2: one reduce task per node ---------------------------------
        live_nodes = [n for n in self.nodes_list if rows_by_node[n]]
        for n in self.nodes_list:
            if not rows_by_node[n]:
                print(f"No successful initialisations for node {n}")
                self.storage[n] = None

        pool = Pool(min(self.cfg.n_process, max(len(live_nodes), 1)))
        async_results = [
            pool.apply_async(self.worker_reduce, kwds={'node': n, 'init_rows': rows_by_node[n]})
            for n in live_nodes
        ]
        pool.close()

        for node, async_result in zip(live_nodes, tqdm(async_results, desc="Selecting per node", unit="node")):
            try:
                bic, aic, returned_node = async_result.get(timeout=self.cfg.timeout_per_node)
            except Exception as exc:                                   # noqa: BLE001
                print(f"Reduce failed for node {node}: {type(exc).__name__}: {exc}")
                self.storage[node] = None
                continue
            self.storage[returned_node] = [bic, aic]

        pool.terminate()
        pool.join()

        return self.storage

    def worker_fit_init(self, node, init):
        from models.global_model.run_experiment_ic import MainLoop as MainLoop
        model_loop = MainLoop(self, node)
        return model_loop.fit_one_init(init)

    def worker_reduce(self, node, init_rows):
        from models.global_model.run_experiment_ic import MainLoop as MainLoop
        model_loop = MainLoop(self, node)
        _holdout, BIC, AIC, node = model_loop.reduce_from_rows(init_rows)
        return BIC, AIC, node

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
