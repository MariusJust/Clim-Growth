"""Panel cluster-bootstrap of the top climate-growth networks.

Refits the selected architectures on ``n_boot`` country-cluster bootstrap
resamples and records the climate response surface each time, yielding
pointwise confidence bands for individual architectures and for an equal-weight
top-model ensemble.

Why country-cluster resampling. The panel has one unit per country with rich
within-country serial dependence and country fixed effects. Drawing **countries
with replacement** (a cluster/pairs bootstrap) respects that dependence and
matches the country-clustered inference standard in the climate-growth
literature (Cameron-Gelbach-Miller). Because the network builds its fixed
effects and country trends internally from the panel columns, a duplicated
country simply becomes a fresh unit with its own FE/trend -- no external trend
bookkeeping is needed. Resampling is done on the *output* of ``Prepare`` so the
input standardization stays fixed at the full-sample moments across resamples.

Scope. The architecture set is held fixed at the selected top models, so the
bands reflect estimation uncertainty over those architectures, not uncertainty
from rerunning model selection inside each bootstrap replication.

Run on the server (needs TensorFlow), Hydra-driven like ``Main.py``:

    PYTHONPATH=src python src/bootstrap.py \
        instance.bootstrap.run_dir="runs/estimation/..." \
        instance.bootstrap.top_k=10 instance.bootstrap.n_boot=500

  - ``instance.bootstrap.nodes``   optional explicit architecture list.
  - ``instance.bootstrap.node``    optional single architecture string.
  - ``instance.bootstrap.run_dir`` optional finished estimation run whose
                                  ``results.npy`` supplies top AIC nodes.
  - ``instance.bootstrap.top_k``   number of AIC-ranked architectures.
  - ``instance.bootstrap.n_boot``  number of resamples.
  - ``instance.bootstrap.n_inits`` random initializations per node/bootstrap.
  - standardization grid, target_mode, country_trends, etc. are inherited from
    the usual ``instance`` config.

Outputs into the Hydra run dir:
  - ``bootstrap_surfaces.npy``       (n_boot, n_nodes, H, W), NaN on failed fits
  - ``bootstrap_draw_indices.npy``   sampled country column indices by rep
  - ``bootstrap_draw_countries.npy`` sampled country labels by rep
  - ``bootstrap_rep_status.csv``     status/loss metadata by rep and node
  - ``bootstrap_bands.npz``          T, P grids + per-node and equal-pooled bands
"""
from utils.miscelaneous import turn_off_warnings
turn_off_warnings()

import os
import ast
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import multiprocessing as mp

from models.helper_functions.global_model import load_data
from models import MultivariateModelGlobal as Model
from utils import create_pred_input
from utils.miscelaneous import Find_data_file
from utils.config import flatten_instance


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def resample_columns(growth, precip, temp, rng):
    """Country-cluster resample: draw the panel columns (countries) with
    replacement and relabel duplicates so each becomes a distinct unit. The
    same column draw is applied to growth, precipitation, and temperature so
    the three stay aligned."""
    cols = list(growth["global"].columns)
    n = len(cols)
    draw = rng.integers(0, n, size=n)
    new_names = [f"{cols[j]}__b{i}" for i, j in enumerate(draw)]

    draw_countries = [str(cols[j]) for j in draw]

    def take(dct):
        df = dct["global"].iloc[:, draw].copy()
        df.columns = new_names
        return {"global": df}

    return take(growth), take(precip), take(temp), draw, draw_countries


def _moving_block_row_order(n_rows, block_len, rng):
    """Moving-block row (time) order: draw ceil(n_rows/block_len) contiguous
    blocks of length block_len with replacement from start positions in
    [0, n_rows-block_len], concatenate, and truncate to n_rows. Each block keeps
    the full cross-section (all countries), so contemporaneous cross-sectional
    dependence is preserved while serial dependence is kept within a block."""
    block_len = max(1, min(int(block_len), int(n_rows)))
    n_blocks = int(np.ceil(n_rows / block_len))
    starts = rng.integers(0, n_rows - block_len + 1, size=n_blocks)
    order = np.concatenate([np.arange(s, s + block_len) for s in starts])[:n_rows]
    return order.astype(int), starts.astype(np.int32)


def resample_residual_block(Yhat, Uhat, template, block_len, rng):
    """Residual moving-block bootstrap over time.

    Holds the design fixed (precipitation, temperature, the panel mask, and the
    fitted FE/trend structure carried by ``Yhat``). Only the idiosyncratic
    residual matrix ``Uhat`` (T x N, NaN at missing cells) is block-resampled
    along the time axis, keeping the full country vector per block, and added
    back to the point-estimate fit::

        y* = Yhat + Uhat[block_order, :]

    Original missingness is preserved wherever ``Yhat`` is NaN; a cell whose
    resampled source is missing becomes NaN (a small, documented sample-size
    drift inherent to block-resampling an unbalanced panel). This scheme,
    unlike the country-cluster pairs bootstrap, does not assume cross-sectional
    independence of countries. Returns the growth dict and the block starts."""
    n_rows = Yhat.shape[0]
    order, starts = _moving_block_row_order(n_rows, block_len, rng)
    Ystar = Yhat + Uhat[order, :]
    df = template.copy()
    df.iloc[:, :] = Ystar
    return {"global": df}, starts


def _point_fit_residuals(node, cfg_dict, growth, precip, temp, fit_kw, seed):
    """Fit the point model once on the original sample and return the full
    fitted-growth and residual matrices (T x N, NaN at missing cells), the base
    for the residual moving-block bootstrap. Mirrors the within reconstruction
    in MultivariateModel.in_sample_predictions: full fit = climate net +
    W @ gamma_hat, with gamma_hat recovered exactly by OLS."""
    import tensorflow as tf

    tf.random.set_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)

    factory = Model(node=node, cfg=cfg_dict, x_train={0: temp, 1: precip}, y_train=growth)
    factory.country_map = None
    factory.Depth = len(node)
    factory.get_model()
    factory.fit(lr=fit_kw["lr"], min_delta=fit_kw["min_delta"],
                patience=fit_kw["patience"], verbose=0)
    factory.in_sample_predictions()

    if not bool(getattr(factory, "within_projection", False)):
        raise RuntimeError("residual_block bootstrap requires within_projection=true")
    proj = factory._get_within_projector()
    y_mat = np.array(factory.y_train_df["global"], dtype=float)
    f_mat = np.array(factory.in_sample_pred["global"], dtype=float)
    y_obs = y_mat[proj.t_arr, proj.n_arr]
    f_obs = f_mat[proj.t_arr, proj.n_arr]
    gamma = proj.recover_gamma(y_obs - f_obs)
    full_obs = f_obs + proj.W @ gamma
    Yhat = np.full_like(y_mat, np.nan)
    Uhat = np.full_like(y_mat, np.nan)
    Yhat[proj.t_arr, proj.n_arr] = full_obs
    Uhat[proj.t_arr, proj.n_arr] = y_obs - full_obs
    return Yhat, Uhat


# --------------------------------------------------------------------------- #
# Worker (one bootstrap replication)
# --------------------------------------------------------------------------- #
_CTX = {}


def _init_worker(nodes, cfg_dict, growth, precip, temp, pred_input,
                 grid_shape, fit_kw, n_inits, scheme, block_len, Yhat, Uhat):
    """Pool initializer: stash the shared, read-only context once per worker
    (spawn pickles these args n_process times, not per replication)."""
    global _CTX
    _CTX = dict(nodes=nodes, cfg_dict=cfg_dict, growth=growth, precip=precip,
                temp=temp, pred_input=pred_input, grid_shape=grid_shape,
                fit_kw=fit_kw, n_inits=n_inits, scheme=scheme,
                block_len=block_len, Yhat=Yhat, Uhat=Uhat)


def _fit_one_init(node, g, p, t, seed):
    import tensorflow as tf

    tf.random.set_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)

    factory = Model(node=node, cfg=_CTX["cfg_dict"], x_train={0: t, 1: p}, y_train=g)
    factory.country_map = None
    factory.Depth = len(node)
    factory.get_model()

    fk = _CTX["fit_kw"]
    factory.fit(lr=fk["lr"], min_delta=fk["min_delta"], patience=fk["patience"],
                verbose=0)

    history = getattr(factory.model, "history", None)
    losses = []
    if history is not None:
        losses = history.history.get("loss", [])
    if not losses:
        raise RuntimeError("fit produced no loss history")

    best_loss = float(np.nanmin(np.asarray(losses, dtype=float)))
    if not np.isfinite(best_loss):
        raise RuntimeError(f"non-finite best loss: {best_loss}")

    Z = factory.model_visual.predict([_CTX["pred_input"]], verbose=0)
    Z = np.asarray(Z, dtype=np.float32).reshape(_CTX["grid_shape"])
    if not np.all(np.isfinite(Z)):
        raise RuntimeError("predicted surface contains non-finite values")

    return best_loss, int(len(losses)), Z


def _run_rep(task):
    rep, seed = task
    rng = np.random.default_rng(seed)

    if _CTX["scheme"] == "residual_block":
        # Residual moving-block over time: precip/temp fixed, growth = Yhat + Uhat*.
        g, starts = resample_residual_block(
            _CTX["Yhat"], _CTX["Uhat"], _CTX["growth"]["global"],
            _CTX["block_len"], rng,
        )
        p, t = _CTX["precip"], _CTX["temp"]
        draw = starts
        draw_countries = np.array([], dtype=object)
    else:
        g, p, t, draw, draw_countries = resample_columns(
            _CTX["growth"], _CTX["precip"], _CTX["temp"], rng
        )

    nodes = _CTX["nodes"]
    rep_surfaces = np.full((len(nodes),) + _CTX["grid_shape"], np.nan, dtype=np.float32)
    rows = []

    for node_idx, node in enumerate(nodes):
        best = None
        errors = []
        for init_idx in range(_CTX["n_inits"]):
            init_seed = int(seed + 100_000 * (node_idx + 1) + init_idx)
            try:
                best_loss, epochs, Z = _fit_one_init(node, g, p, t, init_seed)
                if best is None or best_loss < best[0]:
                    best = (best_loss, init_idx, epochs, Z)
            except Exception as exc:  # keep the pool alive; record the bad fit.
                errors.append(f"init {init_idx}: {type(exc).__name__}: {exc}")

        if best is None:
            rows.append(dict(rep=rep, node_idx=node_idx, node=str(node),
                             ok=False, best_init=np.nan, best_loss=np.nan,
                             epochs=np.nan, error=" | ".join(errors)))
            continue

        best_loss, best_init, epochs, Z = best
        rep_surfaces[node_idx] = Z
        rows.append(dict(rep=rep, node_idx=node_idx, node=str(node),
                         ok=True, best_init=best_init, best_loss=best_loss,
                         epochs=epochs, error=" | ".join(errors)))

    return rep, draw.astype(np.int32), np.asarray(draw_countries, dtype=object), rep_surfaces, rows


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _parse_node(node):
    if isinstance(node, np.ndarray):
        node = node.tolist()
    if isinstance(node, (tuple, list)):
        return tuple(int(x) for x in node)
    return tuple(int(x) for x in ast.literal_eval(str(node)))


def _as_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def _top_nodes_by_aic(run_dir, top_k):
    raw = dict(np.load(Path(run_dir) / "results.npy", allow_pickle=True).item())
    raw = {k: v for k, v in raw.items() if v is not None}
    ranked = sorted(raw, key=lambda n: raw[n][1])  # AIC at index 1
    keep = [
        _parse_node(n) for n in ranked
        if os.path.exists(Path(run_dir) / "parameters" / f"{n}.weights.h5")
    ]
    keep = keep or [_parse_node(n) for n in ranked]
    return keep[:top_k]


def _resolve_nodes(inst, boot):
    nodes = boot.nodes
    if nodes:
        return [_parse_node(n) for n in _as_list(nodes)]

    if boot.node:
        return [_parse_node(boot.node)]

    if boot.run_dir:
        nodes = _top_nodes_by_aic(boot.run_dir, int(boot.top_k))
        if nodes:
            return nodes

    nodes_list = inst.get("nodes_list", None)
    if nodes_list:
        return [_parse_node(n) for n in nodes_list]

    raise SystemExit("Specify instance.bootstrap.nodes, .node, or .run_dir.")


def _nan_bands(stack, axis=0):
    return (
        np.nanmean(stack, axis=axis),
        np.nanmedian(stack, axis=axis),
        np.nanpercentile(stack, 2.5, axis=axis),
        np.nanpercentile(stack, 97.5, axis=axis),
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    inst = flatten_instance(cfg.instance)
    boot = inst.bootstrap
    run_dir = Path(HydraConfig.get().runtime.output_dir)
    t0 = time.time()

    nodes = _resolve_nodes(inst, boot)
    n_boot = int(boot.n_boot)
    n_inits = int(boot.n_inits)
    n_proc = int(boot.n_process) if boot.n_process is not None else int(inst.n_process)
    report_every = int(boot.report_every)
    seed0 = int(inst.seed_value)
    target_mode = inst.target_mode
    print(f"Bootstrap: nodes={nodes}, n_boot={n_boot}, n_inits={n_inits}, "
          f"n_process={n_proc}, target_mode={target_mode}, weights=equal")
    scheme = str(boot.get("scheme", "cluster")).lower()
    block_len = int(boot.get("block_len", 4))
    if scheme == "residual_block":
        print(f"  scheme=residual_block (residual moving-block over time), block_len={block_len}")
    else:
        scheme = "cluster"
        print("  scheme=cluster (country pairs/cluster)")

    # Full-sample prepared data (resampled per replication at column level).
    growth, precip, temp = load_data("IC", inst.data_source,
                                     end_year=inst.get("data_end", None),
                                     target_mode=target_mode)

    # Fixed (T, P) grid + standardization from the full-sample raw moments.
    raw = pd.read_excel(Find_data_file("MainData.xlsx"))
    if inst.get("data_end", None) is not None:
        raw = raw[raw["Year"] <= inst.data_end]
    mean_T, std_T = float(np.nanmean(raw["TempPopWeight"])), float(np.nanstd(raw["TempPopWeight"]))
    mean_P, std_P = float(np.nanmean(raw["PrecipPopWeight"])), float(np.nanstd(raw["PrecipPopWeight"]))
    pred_input, T_grid, P_grid = create_pred_input(
        mc=False, mean_T=mean_T, std_T=std_T, mean_P=mean_P, std_P=std_P,
        precip_capped=True)
    grid_shape = T_grid.shape

    cfg_dict = OmegaConf.to_container(inst, resolve=True)
    fit_kw = dict(lr=float(inst.lr), min_delta=float(inst.min_delta),
                  patience=int(inst.patience))

    # Residual moving-block needs one point fit on the original sample to build
    # the fitted-growth and residual base matrices (Yhat, Uhat).
    Yhat = Uhat = None
    draw_width = n_countries = len(growth["global"].columns)
    if scheme == "residual_block":
        print("  fitting point model on the original sample for the residual base ...")
        Yhat, Uhat = _point_fit_residuals(nodes[0], cfg_dict, growth, precip,
                                          temp, fit_kw, seed0)
        n_rows = int(Yhat.shape[0])
        bl = max(1, min(block_len, n_rows))
        draw_width = int(np.ceil(n_rows / bl))

    tasks = [(b, seed0 + 1 + b) for b in range(n_boot)]
    n_nodes = len(nodes)
    surfaces = np.full((n_boot, n_nodes) + grid_shape, np.nan, dtype=np.float32)
    draw_indices = np.full((n_boot, draw_width), -1, dtype=np.int32)
    draw_countries = np.full((n_boot, n_countries), "", dtype=object)
    status_rows = []

    init_args = (nodes, cfg_dict, growth, precip, temp, pred_input,
                 grid_shape, fit_kw, n_inits, scheme, block_len, Yhat, Uhat)
    done = 0
    with mp.Pool(processes=n_proc, initializer=_init_worker, initargs=init_args) as pool:
        for rep, draw, countries, Z, rows in pool.imap_unordered(_run_rep, tasks):
            draw_indices[rep] = draw
            if scheme != "residual_block":
                draw_countries[rep] = countries
            surfaces[rep] = Z
            status_rows.extend(rows)
            done += 1
            if done % report_every == 0 or done == n_boot:
                ok = int(np.isfinite(surfaces).all(axis=(-2, -1)).sum())
                total = done * n_nodes
                print(f"  {done}/{n_boot} replications done "
                      f"({ok}/{total} node fits ok, "
                      f"{int(time.time()-t0)}s elapsed)")

    # Pointwise bands (raw and mean-centered at the sample-mean grid cell).
    ti = int(np.argmin(np.abs(T_grid[0, :] - mean_T)))
    pi = int(np.argmin(np.abs(P_grid[:, 0] - mean_P)))
    centered = surfaces - surfaces[:, :, pi, ti][:, :, None, None]

    mean_r, med_r, lo_r, hi_r = _nan_bands(surfaces, axis=0)
    mean_c, med_c, lo_c, hi_c = _nan_bands(centered, axis=0)

    pooled = surfaces.reshape(n_boot * n_nodes, *grid_shape)
    pooled_centered = centered.reshape(n_boot * n_nodes, *grid_shape)
    pmean_r, pmed_r, plo_r, phi_r = _nan_bands(pooled, axis=0)
    pmean_c, pmed_c, plo_c, phi_c = _nan_bands(pooled_centered, axis=0)

    np.save(run_dir / "bootstrap_surfaces.npy", surfaces)
    np.save(run_dir / "bootstrap_draw_indices.npy", draw_indices)
    np.save(run_dir / "bootstrap_draw_countries.npy", draw_countries)
    pd.DataFrame(status_rows).sort_values(["rep", "node_idx"]).to_csv(
        run_dir / "bootstrap_rep_status.csv", index=False
    )
    np.savez(run_dir / "bootstrap_bands.npz",
             nodes=np.array([str(n) for n in nodes], dtype=object),
             T=T_grid, P=P_grid,
             mean_by_node=mean_r, median_by_node=med_r,
             lo_by_node=lo_r, hi_by_node=hi_r,
             mean=mean_r, median=med_r, lo=lo_r, hi=hi_r,
             mean_centered=mean_c, median_centered=med_c,
             lo_centered=lo_c, hi_centered=hi_c,
             pooled_mean=pmean_r, pooled_median=pmed_r,
             pooled_lo=plo_r, pooled_hi=phi_r,
             pooled_mean_centered=pmean_c, pooled_median_centered=pmed_c,
             pooled_lo_centered=plo_c, pooled_hi_centered=phi_c,
             center_idx=np.array([pi, ti]),
             scheme=np.array(scheme), block_len=np.array(int(block_len)))

    ok_mask = np.isfinite(surfaces).all(axis=(-2, -1))
    ok_count = int(ok_mask.sum())
    total_count = int(n_boot * n_nodes)
    print(f"Successful node fits: {ok_count}/{total_count}")
    print(f"Saved surfaces -> {run_dir/'bootstrap_surfaces.npy'}")
    print(f"Saved draw indices -> {run_dir/'bootstrap_draw_indices.npy'}")
    print(f"Saved draw countries -> {run_dir/'bootstrap_draw_countries.npy'}")
    print(f"Saved status -> {run_dir/'bootstrap_rep_status.csv'}")
    print(f"Saved bands -> {run_dir/'bootstrap_bands.npz'}")
    print(f"Bootstrap finished in {int(time.time()-t0)}s")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
