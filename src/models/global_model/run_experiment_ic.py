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
    """Fit one architecture over ``no_inits`` initialisations and keep the best by BIC.

    Two execution paths share the same primitives:

      * serial   -- ``run_experiment()`` loops the initialisations in one process.
        This is the original behaviour and the only path used for Monte Carlo
        and for Holdout.
      * parallel -- ``fit_one_init(j)`` is dispatched once per (node, init) pair
        by ``utils.parallel.multiprocess`` and ``reduce_from_rows()`` then selects
        the winner. Requires ``use_diagnostics: true``, because across processes
        the only way to recover a fitted model is to reload its snapshot from
        ``<run_dir>/diagnostics/<node>/init_<j>.weights.h5``.

    Both paths call ``_fit_init`` and ``_finalise``, so the selection logic exists
    in exactly one place and cannot drift between them.
    """

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

    # ------------------------------------------------------------------
    # shared primitives
    # ------------------------------------------------------------------

    def _prepare_factory(self):
        """Point the factory at this node's data. Idempotent."""
        self.factory.x_train = {0: self.temp, 1: self.precip}
        self.factory.y_train = self.growth
        self.factory.node = self.node

    def _diagnostics_dir(self):
        return Path(self.run_dir) / "diagnostics" / str(self.node)

    def _use_diagnostics(self):
        return self.data is None and bool(getattr(self.cfg, "use_diagnostics", False))

    def _fit_init(self, j, use_diagnostics, diagnostics_dir):
        """Fit initialisation ``j``. Returns (model_instance, BIC, AIC, row_or_None).

        The seeding is exactly the original: seed_value + j applied to TensorFlow,
        numpy and the stdlib RNG immediately before the model is constructed.
        """
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

        bic = model_instance.BIC
        aic = model_instance.AIC

        row = None
        if use_diagnostics and history_frame is not None:
            row = save_init_diagnostics(
                model_instance=model_instance,
                history_frame=history_frame,
                diagnostics_dir=diagnostics_dir,
                init_index=j,
                seed_value=current_seed,
                bic=bic,
                aic=aic,
                R2=model_instance.R2['global'] if model_instance.R2 is not None and 'global' in model_instance.R2 else None,
            )

        elapsed = int(time.time() - time_start)
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"Finished node {self.node} with initialization {j+1}/{self.cfg.no_inits} in {hours} hours, {minutes} minutes, and {seconds} seconds")

        return model_instance, bic, aic, row

    def _finalise(self, bic_list, aic_list, init_rows, models, use_diagnostics, diagnostics_dir):
        """Select the best initialisation and write every artefact for this node.

        ``models`` is the in-memory list of fitted models on the serial path and
        None on the parallel path, where the winner must be reloaded from its
        diagnostics snapshot. Selection itself is identical either way: BIC from
        the best-BIC initialisation, AIC from the best-AIC initialisation, chosen
        independently, as in the original implementation.
        """
        bic_arr = np.asarray(bic_list, dtype=float)
        aic_arr = np.asarray(aic_list, dtype=float)

        best_idx_BIC = int(np.nanargmin(bic_arr))
        best_idx_AIC = int(np.nanargmin(aic_arr))

        print(f"saving model parameters to: {self.run_dir}/parameters/{self.node}.weights.h5")

        path = f"{self.run_dir}/parameters/{self.node}.weights.h5"
        os.makedirs(os.path.dirname(path), exist_ok=True)

        best_idx_save = best_idx_BIC

        if use_diagnostics:
            best_snapshot_path = diagnostics_dir / f"init_{best_idx_save}.weights.h5"
            best_model = self._build_model_instance()
            best_model.load_params(str(best_snapshot_path))
            best_model.in_sample_predictions()
            best_model.save_params(path)
        else:
            if models is None:
                raise RuntimeError(
                    "parallel (node, init) execution requires use_diagnostics: true, "
                    "because the winning model is reloaded from its snapshot.")
            best_model = models[best_idx_save]
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
                recorded_aic=aic_arr[best_idx_save],
                recomputed_aic=best_model.AIC,
            )

        return np.nan, bic_arr[best_idx_BIC], aic_arr[best_idx_AIC], self.node

    # ------------------------------------------------------------------
    # parallel entry points, one (node, init) task each
    # ------------------------------------------------------------------

    def fit_one_init(self, j):
        """One (node, init) task. Writes the snapshot and returns its summary row."""
        self._prepare_factory()
        diagnostics_dir = self._diagnostics_dir()
        diagnostics_dir.mkdir(parents=True, exist_ok=True)

        _, bic, aic, row = self._fit_init(j, use_diagnostics=True, diagnostics_dir=diagnostics_dir)
        return float(bic), float(aic), row

    def reduce_from_rows(self, init_rows):
        """Select the winner for this node from the rows its (node, init) tasks returned."""
        self._prepare_factory()
        diagnostics_dir = self._diagnostics_dir()

        rows = sorted([r for r in init_rows if r is not None], key=lambda r: int(r["init"]))
        if not rows:
            raise RuntimeError(f"no successful initialisations for node {self.node}")

        bic_by_idx = {int(r["init"]): float(r["BIC"]) for r in rows}
        aic_by_idx = {int(r["init"]): float(r["AIC"]) for r in rows}
        n = int(self.cfg.no_inits)
        bic_list = [bic_by_idx.get(j, np.nan) for j in range(n)]
        aic_list = [aic_by_idx.get(j, np.nan) for j in range(n)]

        return self._finalise(bic_list, aic_list, rows, models=None,
                              use_diagnostics=True, diagnostics_dir=diagnostics_dir)

    # ------------------------------------------------------------------
    # serial entry point, unchanged behaviour
    # ------------------------------------------------------------------

    def run_experiment(self):
        self._prepare_factory()

        use_diagnostics = self._use_diagnostics()
        diagnostics_dir = self._diagnostics_dir()
        if use_diagnostics:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

        init_rows = []

        #loop over initializations
        for j in range(self.cfg.no_inits):
            model_instance, bic, aic, row = self._fit_init(j, use_diagnostics, diagnostics_dir)

            self.models_tmp[j] = model_instance
            #saves the information criteria
            self.BIC_list[j] = bic
            self.AIC_list[j] = aic
            if row is not None:
                init_rows.append(row)

        # Select the best initialization based on BIC (or AIC)

        #only save the model parameters if the data is the real data, and not simulated data
        if self.data is None:
            return self._finalise(self.BIC_list, self.AIC_list, init_rows,
                                  models=self.models_tmp,
                                  use_diagnostics=use_diagnostics,
                                  diagnostics_dir=diagnostics_dir)
        else: #Monte carlo simulation
            best_idx_BIC = int(np.argmin(self.BIC_list))
            best_idx_AIC = int(np.argmin(self.AIC_list))
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
