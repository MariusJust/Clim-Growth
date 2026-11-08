"""Run pass 2 of an IC sweep offline, from the diagnostics snapshots on disk.

Why this exists
---------------
Run 2026-09-28_20-15-05_global_IC_country_trends_levels completed 2111 of 2160
(node, init) fits in pass 1 and was then killed: the job allocated 125G for 55
TensorFlow workers wanting 2.0-3.0G each, so the cgroup OOM killer removed
workers, and multiprocessing.Pool replaces a dead worker without re-dispatching
its task. Pass 2 never started, so there is no results.npy and no parameters/.

Every completed fit is on disk as diagnostics/<node>/init_<j>.weights.h5. This
script reloads each snapshot, recomputes its information criteria with
in_sample_predictions(), rebuilds the per-initialisation row that
save_init_diagnostics() would have returned, and hands the rows to
MainLoop.reduce_from_rows(). Nothing is refitted.

Unequal initialisation counts are handled by the existing code: reduce_from_rows
fills absent inits with NaN and _finalise selects with nanargmin. The 21
architectures short of 40 are all three layer (34 to 39 inits); every one and
two layer architecture has the full 40.

Round-trip check
----------------
On the normal path, summary.csv compares the AIC recorded during training with
the AIC recomputed from the reloaded snapshot. Here both numbers come from the
reload, so that check is vacuous and is not evidence of anything. In its place
this script reports, per architecture, the Spearman correlation between the
recomputed BIC across initialisations and loss_best read from the training
history CSVs. Those histories were written during training and never reloaded,
so a high correlation is independent evidence that load_params restores the
trained weights faithfully. A correlation near zero means the round trip is
broken and the output must not be used.

Usage
-----
    python scripts/2026-10-02-reduce-from-snapshots.py \
        --run-dir runs/estimation/2026-09-28_20-15-05_global_IC_country_trends_levels

Writes, under the run directory:
    parameters/<node>.weights.h5 and the FE / trend CSVs
    diagnostics/<node>/initialisations.csv and summary.csv
    results.npy
    reduce_check.csv
Refuses to start if results.npy already exists, unless --force is passed.
"""

from __future__ import annotations

import argparse
import ast
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def spearman(a, b):
    """Rank correlation without a scipy dependency. NaN if degenerate."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return np.nan
    ra = pd.Series(a[ok]).rank().to_numpy()
    rb = pd.Series(b[ok]).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def load_cfg(run_dir):
    from omegaconf import OmegaConf
    from utils.config import flatten_instance

    for rel in ("config/config.yaml", ".hydra/config.yaml"):
        path = run_dir / rel
        if path.exists():
            cfg = OmegaConf.load(path)
            if "instance" in cfg:
                return flatten_instance(cfg.instance), path
            return flatten_instance(cfg), path
    raise SystemExit(f"no config found under {run_dir}/config or {run_dir}/.hydra")


class ParentShim:
    """The three attributes MainLoop.__init__ reads off its parent."""

    def __init__(self, cfg, run_dir):
        self.cfg = cfg
        self.data = None
        self.run_dir = str(run_dir)


def history_stats(diag_dir, j):
    """(epochs, loss_final, loss_best, loss_best_epoch, lr_final) from the CSV."""
    from models.helper_functions.global_model.diagnostics import _series_stats

    path = diag_dir / f"init_{j}_history.csv"
    if not path.exists():
        return -1, np.nan, np.nan, -1, np.nan
    frame = pd.read_csv(path)
    loss_final, loss_best, loss_best_epoch = _series_stats(frame, "loss")
    lr_final, _, _ = _series_stats(frame, "learning_rate")
    if np.isnan(lr_final):
        lr_final, _, _ = _series_stats(frame, "lr")
    return int(len(frame)), loss_final, loss_best, loss_best_epoch, lr_final


def score_node(loop, node, diag_dir, no_inits, seed_value):
    """Reload every snapshot for one architecture and rebuild its rows."""
    loop.node = node
    loop._prepare_factory()

    model = loop._build_model_instance()
    rows = []

    for j in range(no_inits):
        snapshot = diag_dir / f"init_{j}.weights.h5"
        if not snapshot.exists():
            continue
        try:
            model.load_params(str(snapshot))
            model.in_sample_predictions()
        except Exception as exc:                                    # noqa: BLE001
            print(f"    init {j}: could not reload ({type(exc).__name__}: {exc})")
            continue

        epochs, loss_final, loss_best, loss_best_epoch, lr_final = history_stats(diag_dir, j)
        r2 = model.R2.get("global") if isinstance(model.R2, dict) else None

        rows.append({
            "init": int(j),
            "seed": int(seed_value + j),
            "epochs": epochs,
            "loss_final": loss_final,
            "loss_best": loss_best,
            "loss_best_epoch": loss_best_epoch,
            "lr_final": lr_final,
            "BIC": float(model.BIC),
            "AIC": float(model.AIC),
            "R2": float(r2) if r2 is not None else np.nan,
            "m_effective": int(getattr(model, "m_effective", -1) or -1),
            "n_effective": float(getattr(model, "n_effective", np.nan)),
            "snapshot": snapshot.name,
        })

    return rows


def main():
    import tensorflow as tf
    from models.global_model.run_experiment_ic import MainLoop
    from utils import save_numpy

    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing results.npy")
    ap.add_argument("--check-data", action="store_true",
                    help="print a fingerprint of the loaded panel and exit. Run "
                         "this on the server and on the local machine and compare: "
                         "the recomputed information criteria are only valid if "
                         "both sides load an identical panel.")
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not (run_dir / "diagnostics").is_dir():
        raise SystemExit(f"{run_dir}/diagnostics does not exist")

    results_path = run_dir / "results.npy"
    if results_path.exists() and not args.force:
        raise SystemExit(f"{results_path} already exists; pass --force to overwrite")

    cfg, cfg_path = load_cfg(run_dir)
    print(f"config      {cfg_path}")
    print(f"run_dir     {run_dir}")

    nodes = [ast.literal_eval(s) for s in cfg.nodes_list]
    no_inits = int(cfg.no_inits)
    seed_value = int(cfg.seed_value)
    print(f"{len(nodes)} architectures, up to {no_inits} initialisations each\n")

    parent = ParentShim(cfg, run_dir)
    loop = MainLoop(parent, nodes[0])

    if args.check_data:
        import hashlib
        import keras
        print(f"tensorflow  {tf.__version__}")
        print(f"keras       {keras.__version__}")
        print(f"data_source {cfg.data_source}")
        print(f"data_end    {getattr(cfg, 'data_end', None)}")
        print(f"target_mode {getattr(cfg, 'target_mode', 'growth')}")
        for name, frame in (("target", loop.growth),
                            ("precip", loop.precip),
                            ("temp", loop.temp)):
            arr = np.ascontiguousarray(np.asarray(frame, dtype="float64"))
            digest = hashlib.md5(np.nan_to_num(arr, nan=-9e99).tobytes()).hexdigest()
            print(f"{name:<11} shape={arr.shape}  "
                  f"nansum={np.nansum(arr):.10g}  md5={digest[:16]}")
        return

    storage = {}
    checks = []
    t0 = time.time()

    for i, node in enumerate(nodes, 1):
        diag_dir = run_dir / "diagnostics" / str(node)
        if not diag_dir.is_dir():
            print(f"[{i:>2}/{len(nodes)}] {str(node):<16} no diagnostics directory, skipped")
            storage[node] = None
            continue

        rows = score_node(loop, node, diag_dir, no_inits, seed_value)
        if not rows:
            print(f"[{i:>2}/{len(nodes)}] {str(node):<16} no reloadable snapshots, skipped")
            storage[node] = None
            continue

        _holdout, bic, aic, returned = loop.reduce_from_rows(rows)
        storage[returned] = [bic, aic]

        rho = spearman([r["BIC"] for r in rows], [r["loss_best"] for r in rows])
        best = min(rows, key=lambda r: r["BIC"])
        checks.append({
            "node": str(node),
            "n_inits": len(rows),
            "best_init": best["init"],
            "BIC": bic,
            "AIC": aic,
            "R2_best": best["R2"],
            "m_effective": best["m_effective"],
            "spearman_BIC_vs_loss_best": rho,
        })

        flag = "" if (np.isnan(rho) or rho > 0.8) else "   <-- CHECK"
        print(f"[{i:>2}/{len(nodes)}] {str(node):<16} "
              f"n={len(rows):>2}  BIC={bic:>14.3f}  rho={rho:>6.3f}{flag}")

        # Release this architecture's graphs before building the next one.
        # Without this the script accumulates 54 Keras graphs, which is the
        # same leak that sank the original job.
        tf.keras.backend.clear_session()

    save_numpy(str(results_path), storage)

    check_frame = pd.DataFrame(checks).sort_values("BIC")
    check_frame.to_csv(run_dir / "reduce_check.csv", index=False)

    elapsed = int(time.time() - t0)
    print(f"\nwrote {results_path}")
    print(f"wrote {run_dir / 'reduce_check.csv'}")
    print(f"finished in {elapsed // 3600}h {(elapsed % 3600) // 60}m {elapsed % 60}s\n")

    print("BIC ranking, best 10 architectures")
    cols = ["node", "n_inits", "best_init", "BIC", "AIC", "R2_best",
            "spearman_BIC_vs_loss_best"]
    print(check_frame[cols].head(10).to_string(index=False))

    low = check_frame["spearman_BIC_vs_loss_best"]
    bad = check_frame[low.notna() & (low <= 0.8)]
    if not bad.empty:
        print(f"\nWARNING: {len(bad)} architectures have a weak rank correlation "
              f"between recomputed BIC and the training loss recorded during "
              f"fitting. Inspect those before trusting the selection:")
        print(bad[["node", "n_inits", "spearman_BIC_vs_loss_best"]].to_string(index=False))


if __name__ == "__main__":
    main()
