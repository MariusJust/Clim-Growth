"""Post-hoc Network Information Criterion (NIC) for the climate-growth networks.

Implements the model-selection criterion of Murata, Yoshizawa & Amari (1994),
"Network Information Criterion -- Determining the Number of Hidden Units for an
Artificial Neural Network Model," IEEE Trans. Neural Networks 5(6):865-872.

Motivation
----------
The classical AIC penalty ``2m`` charges every trainable weight as one degree of
freedom. For a neural network the *effective* number of parameters is typically
far smaller than the nominal weight count (the model is over-parameterized /
singular), so a nominal-``m`` penalty over-penalizes capacity and biases
selection toward small networks. NIC generalizes AIC to such *misspecified*
("unfaithful") models by replacing ``m`` with an effective complexity

        p* = tr( I_hat  J_hat^{-1} ),

where ``J_hat`` is the (mean) Hessian of the loss at the fitted weights and
``I_hat`` is the empirical Fisher (mean outer product of the per-observation
loss gradients). When the model is correctly specified and homoskedastic,
``I_hat = J_hat`` and ``p* = m`` -- NIC collapses to AIC. The departure of
``p*`` from ``m`` is exactly the effective-DoF correction.

Squared-error implementation (TIC / Gauss-Newton form)
------------------------------------------------------
With per-observation loss ``l_i = r_i^2 / (2 sigma^2)``, residual
``r_i = y_i - yhat_i`` and ``g_i = d yhat_i / d theta``:

    d l_i / d theta            = -(r_i / sigma^2) g_i
    I_hat = (1/n) sum (dl_i)(dl_i)'   = (1 / (n sigma^4)) sum r_i^2 g_i g_i'
    J_hat = (1/n) sum d^2 l_i         ~ (1 / (n sigma^2)) sum g_i g_i'   (Gauss-Newton:
                                        drops the r_i * d^2 yhat_i term, which has
                                        mean zero at a correctly-located optimum and
                                        is the standard, numerically-stable choice)

so, with sigma^2 estimated by the MSE,

    p* = tr( I_hat J_hat^{-1} )
       = (1/sigma^2) * tr[ (J' diag(r^2) J) (J' J)^{-1} ],

the heteroskedasticity-robust ("sandwich") effective parameter count, where
``J`` is the (n x P) Jacobian of the fitted values w.r.t. the parameters. This is
the Takeuchi (1976) information criterion, which is the operational NIC for the
squared-error loss; we cite Murata et al. (1994) as the neural-network
formulation. If ``r_i^2 = sigma^2`` for all i, then ``p* = P`` and NIC = AIC.

Singularity
-----------
Neural networks are singular models, so ``J'J`` is typically rank-deficient.
We invert ``J'J + ridge * mean(diag(J'J)) * I`` (a small relative ridge) and
report ``p*`` for a sweep of ridge values so its stability can be inspected.

Scope
-----
NIC is used here for *ranking, weighting, and reporting* only. The reported
response surface remains the equal-weight mean of the top models (now the
top set by NIC). This script does not regenerate any surface.

Run on the server (needs TensorFlow), Hydra-driven like ``Main.py`` /
``bootstrap.py``. It refits nothing: it loads each saved architecture's weights
and differentiates the fitted model.

    PYTHONPATH=src python src/network_information_criterion.py \
        instance.nic.run_dir="results/paper/weights/global_model" \
        instance.nic.top_k=20

Config keys (all optional, under ``instance.nic``; sensible fallbacks):
  - run_dir       finished estimation run dir containing results.npy and
                  parameters/<node>.weights.h5. Defaults to the active Hydra run.
  - nodes / node  explicit architecture(s); otherwise all nodes with saved
                  weights in run_dir are used.
  - top_k         if set, restrict the NIC-weight normalization summary to the
                  top_k AIC-ranked nodes (all nodes are still scored).
  - ridge_grid    relative ridge values for the (J'J) inverse sweep.
                  Default [1e-8, 1e-6, 1e-4, 1e-2]; the first is the headline.
  - jac_batch     observations per Jacobian batch (memory control).
  - jac_subsample if >0, randomly subsample this many observations for the
                  moment matrices (unbiased; for very large panels).
  - param_scope   'all' (every trainable weight; default) or 'network'
                  (climate sub-network weights only -- the architecture-varying
                  block; the common FE/trend block then cancels in Delta-NIC).

Outputs into the run dir:
  - nic_results.csv   per node: n, m (nominal), MSE, AIC, p_star (per ridge),
                      NIC (headline ridge), dNIC, nic_akaike_weight, aic rank,
                      nic rank.
  - nic_summary.txt   human-readable ranking + top-k NIC-Akaike weights.
"""
from utils.miscelaneous import turn_off_warnings
turn_off_warnings()

import os
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from models.helper_functions.global_model import load_data
from models.helper_functions.shared import fid_country_map
from models import MultivariateModelGlobal as Model
from utils.config import flatten_instance

# Repo root = parent of src/. Used to resolve run dirs against the project,
# NOT against Hydra's per-run working directory (Hydra chdir's into its run dir).
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# Run-dir resolution
# --------------------------------------------------------------------------- #
def _resolve_run_dir(run_dir_cfg):
    """Locate a run dir containing parameters/ or results.npy. Accepts an
    absolute path, a path relative to the repo root, or a bare run name that is
    searched under runs/estimation/ and results/paper/weights/."""
    if run_dir_cfg:
        p = Path(str(run_dir_cfg))
        if p.is_absolute():
            candidates = [p]
        else:
            candidates = [
                PROJECT_ROOT / p,
                PROJECT_ROOT / "runs" / "estimation" / p,
                PROJECT_ROOT / "results" / "paper" / "weights" / p,
                Path.cwd() / p,
            ]
    else:
        candidates = [PROJECT_ROOT / "results" / "paper" / "weights" / "global_model"]

    for c in candidates:
        if (c / "parameters").is_dir() or (c / "results.npy").exists():
            return c
    tried = "\n  ".join(str(c) for c in candidates)
    raise SystemExit(
        "Could not locate a run dir with parameters/ or results.npy.\n"
        f"Set instance.nic.run_dir (absolute, repo-relative, or a bare run name).\nTried:\n  {tried}"
    )


# --------------------------------------------------------------------------- #
# Node parsing / discovery
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
    return list(value)


def _discover_nodes(run_dir):
    """All nodes with saved weights, AIC-ranked when results.npy is present."""
    run_dir = Path(run_dir)
    pdir = run_dir / "parameters"
    saved = sorted(p.stem.replace(".weights", "") for p in pdir.glob("*.weights.h5"))
    aic_by_node = {}
    res = run_dir / "results.npy"
    if res.exists():
        raw = dict(np.load(res, allow_pickle=True).item())
        for k, v in raw.items():
            if v is not None:
                try:
                    aic_by_node[str(k)] = float(v[1])  # AIC at index 1
                except (TypeError, IndexError):
                    pass
    nodes = [s for s in saved]
    nodes.sort(key=lambda s: aic_by_node.get(s, np.inf))
    return [_parse_node(s) for s in nodes], aic_by_node


def _resolve_nodes(inst, nic, run_dir):
    if nic.get("nodes", None):
        return [_parse_node(n) for n in _as_list(nic.nodes)], {}
    if nic.get("node", None):
        return [_parse_node(nic.node)], {}
    return _discover_nodes(run_dir)


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _build_loaded_model(node, cfg, x_train, y_train, country_map, weights_path):
    """Rebuild one architecture and load its saved weights (no refit)."""
    factory = Model(node=node, cfg=cfg, x_train=x_train, y_train=y_train)
    factory.country_map = country_map
    factory.Depth = len(node)
    inst = factory.get_model()
    inst.model.load_weights(str(weights_path))
    return inst


def _observed_pred_and_resid(inst):
    """Differentiable observed-prediction vector and (numpy) residuals,
    matching exactly the SSE/MSE used by ``in_sample_predictions`` / AIC.

    Returns (pred_fn, resid, n_obs, mse) where pred_fn() recomputes the
    observed prediction vector inside a GradientTape."""
    N = inst.N["global"]
    y = np.asarray(inst.y_train["global"], dtype=np.float64)          # (time, N)
    observed = ~np.isnan(y.reshape(-1))                               # bool (time*N,)
    y_obs = y.reshape(-1)[observed]
    inputs = [inst.input_data[var] for var in inst.input_vars]

    def net_vector():
        preds = inst.model(inputs)                                    # (1, time, >=N)
        pv = tf.reshape(preds[0, :, 0:N], [-1])                       # (time*N,)
        return tf.boolean_mask(pv, observed)

    n_obs = int(inst.noObs["global"])

    if bool(getattr(inst, "within_projection", False)):
        # Within run: the model output is the climate net f. The effective DoF
        # must be measured on the PROJECTED objective ||M(y - f)||^2, so the
        # prediction whose Jacobian we take is M f, and the residual is M(y - f).
        proj = inst._get_within_projector()
        W_tf = tf.constant(proj.W, dtype=tf.float32)
        B_tf = tf.constant(proj.B, dtype=tf.float32)

        def pred_vector():
            f = tf.cast(net_vector(), tf.float32)                          # model casts to float64
            return f - tf.linalg.matvec(W_tf, tf.linalg.matvec(B_tf, f))   # M f

        net0 = net_vector().numpy().astype(np.float64)
        resid = proj.annihilate(y_obs) - proj.annihilate(net0)             # M(y - f)
        mse = float(np.sum(resid ** 2) / n_obs)
        return pred_vector, resid, n_obs, mse

    pred0 = net_vector().numpy().astype(np.float64)
    resid = y_obs - pred0
    mse = float(np.nansum(resid ** 2) / n_obs)
    return net_vector, resid, n_obs, mse


# --------------------------------------------------------------------------- #
# Jacobian + effective parameters
# --------------------------------------------------------------------------- #
def _select_variables(inst, param_scope):
    tv = list(inst.model.trainable_variables)
    if param_scope == "network":
        # climate sub-network weights only: Dense kernels/biases of the MLP.
        # The FE/time-FE/trend block is common across architectures and cancels
        # in Delta-NIC; restricting here improves conditioning.
        sel = [v for v in tv if any(t in v.name.lower()
               for t in ("dense", "kernel", "hidden", "output"))
               and "embedding" not in v.name.lower()]
        return sel or tv
    return tv


def _backing_tf_var(v):
    """Keras 3 wraps each weight in a keras.Variable whose ``.dtype`` is a plain
    string; tf.autodiff needs the underlying tf.Variable. Fall back to ``v`` if
    it is already a tf.Variable."""
    return getattr(v, "_value", None) if getattr(v, "_value", None) is not None else v


def _jacobian(pred_vector, variables, n_obs, jac_batch=None):
    """(n_obs x P) Jacobian of the observed predictions w.r.t. ``variables``,
    computed with **forward-mode** autodiff (one JVP per parameter direction).

    Why forward mode: the panel within-transform (Matrixize) uses a BiasAdd in
    NCHW layout whose reverse-mode gradient (``BiasAddGrad``) is not convertible
    by ``pfor`` and is prohibitively slow row-by-row. Forward mode never builds
    ``BiasAddGrad``; each call propagates one parameter-space tangent through the
    forward graph and yields a full Jacobian column (all n_obs rows at once).
    Cost is P forward passes, cheap for the climate sub-network weights
    (``param_scope='network'``), the only block that varies across architectures
    and the only one needed for the NIC weights (the common FE/trend block
    cancels in Delta-NIC).
    """
    tfvars = [_backing_tf_var(v) for v in variables]
    cols = int(sum(int(np.prod(v.shape)) for v in tfvars))
    J = np.empty((n_obs, cols), dtype=np.float64)
    zeros = [tf.zeros_like(v) for v in tfvars]
    c = 0
    for vi, v in enumerate(tfvars):
        size = int(np.prod(v.shape))
        for k in range(size):
            tangents = list(zeros)
            tangents[vi] = tf.reshape(
                tf.one_hot(k, size, dtype=v.dtype), v.shape)
            with tf.autodiff.ForwardAccumulator(tfvars, tangents) as acc:
                out = pred_vector()                  # (n_obs,)
            jvp = acc.jvp(out)
            if jvp is None:                          # direction not connected
                J[:, c] = 0.0
            else:
                J[:, c] = np.asarray(jvp, dtype=np.float64).reshape(-1)
            c += 1
    return J


def _effective_params(J, resid, mse, ridge_grid):
    """p* = (1/sigma^2) tr[(J' diag(r^2) J)(J'J + ridge)^{-1}] for each ridge.

    Returns a dict {ridge: p_star}. Reduces to P (nominal) under homoskedasticity.
    """
    n, P = J.shape
    r2 = resid ** 2
    A = (J.T * 1.0) @ J / n                       # (1/n) J'J        (P x P)
    B = (J.T * r2) @ J / n                         # (1/n) J' diag(r^2) J
    scale = float(np.mean(np.diag(A))) or 1.0
    out = {}
    for lam in ridge_grid:
        A_reg = A + lam * scale * np.eye(P)
        # tr(B A_reg^{-1}) / sigma^2 ; solve instead of explicit inverse
        M = np.linalg.solve(A_reg, B.T).T          # B A_reg^{-1}
        out[float(lam)] = float(np.trace(M) / mse)
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    inst_cfg = flatten_instance(cfg.instance)
    nic = inst_cfg.get("nic", {})

    run_dir = _resolve_run_dir(nic.get("run_dir", None))
    print(f"[nic] using run_dir: {run_dir}")
    ridge_grid = list(nic.get("ridge_grid", [1e-8, 1e-6, 1e-4, 1e-2]))
    jac_batch = int(nic.get("jac_batch", 512))
    jac_subsample = int(nic.get("jac_subsample", 0))
    param_scope = str(nic.get("param_scope", "network"))
    top_k = nic.get("top_k", None)
    headline_ridge = float(ridge_grid[0])

    nodes, aic_by_node = _resolve_nodes(inst_cfg, nic, run_dir)
    if not nodes:
        raise SystemExit(f"No architectures found under {run_dir}.")
    n_weight_files = len(list((run_dir / "parameters").glob("*.weights.h5")))
    print(f"[nic] scoring {len(nodes)} architecture(s); "
          f"{n_weight_files} weight file(s) in {run_dir}/parameters", flush=True)

    # data (identical pipeline to estimation)
    growth, precip, temp = load_data(
        "IC", inst_cfg.data_source, end_year=inst_cfg.data_end,
        target_mode=inst_cfg.get("target_mode", "growth"),
    )
    country_map = fid_country_map() if str(inst_cfg.data_source).lower() == "ee" else None
    x_train = {0: temp, 1: precip}

    rng = np.random.default_rng(int(inst_cfg.get("seed_value", 0)))
    rows = []
    for node in nodes:
        wpath = run_dir / "parameters" / f"{node}.weights.h5"
        if not wpath.exists():
            print(f"[skip] no weights for {node}")
            continue
        tf.keras.backend.clear_session()
        inst = _build_loaded_model(node, inst_cfg, x_train, growth, country_map, wpath)
        inst.in_sample_predictions()                         # sets MSE/AIC/m/R2
        m_nominal = int(inst.m)
        aic = float(inst.AIC)
        r2 = float(inst.R2["global"]) if inst.R2 and "global" in inst.R2 else np.nan

        pred_vector, resid, n_obs, mse = _observed_pred_and_resid(inst)

        # optional row subsample for the moment matrices (unbiased)
        if jac_subsample and jac_subsample < n_obs:
            idx = np.sort(rng.choice(n_obs, size=jac_subsample, replace=False))
        else:
            idx = None

        variables = _select_variables(inst, param_scope)
        J = _jacobian(pred_vector, variables, n_obs, jac_batch)
        if idx is not None:
            J_use, r_use = J[idx], resid[idx]
        else:
            J_use, r_use = J, resid

        p_stars = _effective_params(J_use, r_use, mse, ridge_grid)
        p_star = p_stars[headline_ridge]
        nic_val = n_obs * np.log(mse) + 2.0 * p_star

        row = dict(node=str(node), n=n_obs, m_nominal=m_nominal, MSE=mse,
                   R2=r2, AIC=aic, p_star=p_star, NIC=nic_val,
                   P_jacobian=J.shape[1], param_scope=param_scope)
        for lam, ps in p_stars.items():
            row[f"p_star_ridge_{lam:.0e}"] = ps
        rows.append(row)
        print(f"[ok] {str(node):<14} m={m_nominal:<5d} p*={p_star:8.2f} "
              f"AIC={aic:12.2f} NIC={nic_val:12.2f}")

    df = pd.DataFrame(rows)
    # Delta-NIC and NIC-Akaike weights (over all scored nodes)
    df = df.sort_values("NIC").reset_index(drop=True)
    df["dNIC"] = df["NIC"] - df["NIC"].min()
    w = np.exp(-0.5 * df["dNIC"].to_numpy())
    df["nic_akaike_weight"] = w / w.sum()
    df["nic_rank"] = np.arange(1, len(df) + 1)
    df["aic_rank"] = df["AIC"].rank(method="min").astype(int)

    out_csv = run_dir / "nic_results.csv"
    df.to_csv(out_csv, index=False)

    # summary
    k = int(top_k) if top_k else len(df)
    head = df.head(k)
    w_top = head["nic_akaike_weight"] / head["nic_akaike_weight"].sum()
    lines = [
        f"NIC results for {run_dir}",
        f"param_scope={param_scope}  headline_ridge={headline_ridge:.0e}  "
        f"ridge_grid={ridge_grid}",
        f"nodes scored: {len(df)}",
        "",
        f"{'rank':>4} {'node':<14} {'m':>5} {'p*':>8} {'AIC':>12} "
        f"{'NIC':>12} {'dNIC':>8} {'w_NIC':>8}",
    ]
    for _, r in df.iterrows():
        lines.append(f"{int(r.nic_rank):>4} {r.node:<14} {int(r.m_nominal):>5} "
                     f"{r.p_star:>8.2f} {r.AIC:>12.2f} {r.NIC:>12.2f} "
                     f"{r.dNIC:>8.2f} {r.nic_akaike_weight:>8.4f}")
    lines += ["", f"Top-{k} NIC-Akaike weights (renormalized within top-{k}):"]
    for (_, r), wt in zip(head.iterrows(), w_top):
        lines.append(f"  {r.node:<14} {wt:>8.4f}")
    summary = "\n".join(lines)
    (run_dir / "nic_summary.txt").write_text(summary, encoding="utf-8")
    print("\n" + summary)
    print(f"\nWrote {out_csv} and {run_dir / 'nic_summary.txt'}")


if __name__ == "__main__":
    main()
