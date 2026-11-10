"""Per-initialisation training diagnostics for the global model.

Reconstructed 2026-09-01. The original file was lost in the OneDrive -> C:\\dev
move (only ``__pycache__/diagnostics.cpython-312.pyc`` survived, and 3.12
bytecode cannot be unmarshalled by an older interpreter). It is rebuilt from its
call sites in ``models/global_model/run_experiment_ic.py``, which pin the
contract exactly:

  * ``save_init_diagnostics`` MUST write ``<diagnostics_dir>/init_<i>.weights.h5``.
    That is not just logging -- after the initialisation loop the selected
    snapshot is reloaded from that exact path::

        best_snapshot_path = diagnostics_dir / f"init_{best_idx_save}.weights.h5"
        best_model.load_params(str(best_snapshot_path))

    so if the file is absent the run fails *after* training, having burnt the
    whole node. It returns one row per initialisation.

  * ``save_summary_diagnostics`` writes the collected rows and compares the AIC
    recorded during training with the AIC recomputed from the reloaded snapshot.
    A mismatch means the save/restore round-trip is not faithful, which would
    silently invalidate model selection.

Everything here is best-effort: a diagnostics failure must never kill a
multi-hour estimation, so the writers swallow their own errors and warn.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Treat |recorded - recomputed| above this as a real snapshot round-trip failure.
AIC_TOL = 1e-6


def _series_stats(history_frame, column):
    """(final, best, epoch_of_best) for a history column, or (nan, nan, -1)."""
    if history_frame is None or column not in history_frame:
        return np.nan, np.nan, -1
    s = pd.to_numeric(history_frame[column], errors="coerce").dropna()
    if s.empty:
        return np.nan, np.nan, -1
    return float(s.iloc[-1]), float(s.min()), int(s.idxmin())


def save_init_diagnostics(model_instance, history_frame, diagnostics_dir,
                          init_index, seed_value, bic, aic, R2):
    """Snapshot one initialisation and return its diagnostics row.

    The weight snapshot is the load-bearing part; the CSV/row are informational.
    """
    diagnostics_dir = _as_path(diagnostics_dir)
    snapshot = diagnostics_dir / f"init_{init_index}.weights.h5"

    # 1. the snapshot the caller will reload. Deliberately NOT wrapped in a
    #    try/except: if this cannot be written the run must fail here, loudly,
    #    rather than after training when load_params hits a missing file.
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    model_instance.save_params(str(snapshot))

    loss_final, loss_best, loss_best_epoch = _series_stats(history_frame, "loss")
    lr_final, _, _ = _series_stats(history_frame, "learning_rate")
    if np.isnan(lr_final):
        lr_final, _, _ = _series_stats(history_frame, "lr")

    row = {
        "init": int(init_index),
        "seed": int(seed_value),
        "epochs": int(len(history_frame)) if history_frame is not None else -1,
        "loss_final": loss_final,
        "loss_best": loss_best,
        "loss_best_epoch": loss_best_epoch,
        "lr_final": lr_final,
        "BIC": float(bic) if bic is not None else np.nan,
        "AIC": float(aic) if aic is not None else np.nan,
        "R2": float(R2) if R2 is not None else np.nan,
        "m_effective": int(getattr(model_instance, "m_effective", -1) or -1),
        "n_effective": float(getattr(model_instance, "n_effective", np.nan)),
        "snapshot": snapshot.name,
    }

    # 2. informational: the full loss path for this initialisation.
    try:
        if history_frame is not None:
            history_frame.to_csv(diagnostics_dir / f"init_{init_index}_history.csv",
                                 index_label="epoch")
    except Exception as exc:                                        # noqa: BLE001
        warnings.warn(f"could not write init_{init_index}_history.csv: {exc}")

    return row


def save_summary_diagnostics(diagnostics_dir, init_rows, best_idx,
                             recorded_aic, recomputed_aic):
    """Write the per-initialisation table and check the snapshot round-trip.

    ``recorded_aic``   AIC stored while that initialisation was trained.
    ``recomputed_aic`` AIC after reloading the snapshot and re-predicting.
    They must agree; a gap means ``save_params``/``load_params`` did not restore
    the same model, so the architecture ranking cannot be trusted.
    """
    diagnostics_dir = _as_path(diagnostics_dir)
    try:
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(list(init_rows))
        if not df.empty:
            df["selected"] = df["init"] == int(best_idx)
            df = df.sort_values("init")
            df.to_csv(diagnostics_dir / "initialisations.csv", index=False)

        rec = float(recorded_aic) if recorded_aic is not None else np.nan
        recomp = float(recomputed_aic) if recomputed_aic is not None else np.nan
        diff = abs(rec - recomp) if np.isfinite(rec) and np.isfinite(recomp) else np.nan
        ok = bool(np.isfinite(diff) and diff <= AIC_TOL)

        pd.DataFrame([{
            "best_init": int(best_idx),
            "aic_recorded": rec,
            "aic_recomputed": recomp,
            "abs_diff": diff,
            "snapshot_roundtrip_ok": ok,
            "tol": AIC_TOL,
            "n_inits": int(len(df)) if not df.empty else 0,
        }]).to_csv(diagnostics_dir / "summary.csv", index=False)

        if not ok:
            warnings.warn(
                f"snapshot round-trip mismatch in {diagnostics_dir.name}: "
                f"AIC recorded {rec:.6f} vs recomputed {recomp:.6f} "
                f"(|diff| {diff:.3g} > {AIC_TOL:g}). Model selection for this "
                f"architecture may not reflect the saved weights.")
    except Exception as exc:                                        # noqa: BLE001
        warnings.warn(f"could not write summary diagnostics: {exc}")


def _as_path(p):
    from pathlib import Path
    return p if hasattr(p, "mkdir") else Path(str(p))
