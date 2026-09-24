"""Regression test for the (node, init) parallelisation.

The refactor moves each initialisation into its own process. That is only safe if
the seeding is history independent: initialisation j must produce the same fitted
model whether it runs in a fresh process or after initialisations 0..j-1 in a
shared one. ``tf.random.set_seed`` is called immediately before the model is
built, so it should be, but "should be" is not evidence.

This script supplies the evidence. Run the SAME configuration twice, once with
``parallel_over_inits: false`` and once with ``true``, then compare.

    python scripts/2026-09-24-init-parallel-regression.py \
        --serial   runs/estimation/<serial_run> \
        --parallel runs/estimation/<parallel_run>

Optionally also check the selected BIC against a pre-refactor run:

        --reference runs/estimation/2026-09-08_12-44-40_global_dynamic

PASS requires every per-initialisation BIC and AIC to agree to the last bit
(max absolute difference exactly 0.0), and the per-node selected values in
results.npy to agree likewise. Anything else is a FAIL and the pilot must not be
dispatched until it is understood.

Exit code 0 on pass, 1 on fail.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _resolve(p: str) -> Path:
    q = Path(p)
    return q if q.is_absolute() else (ROOT / q)


def load_inits(run_dir: Path) -> dict:
    """{node_str: DataFrame indexed by init} from every diagnostics/<node>/initialisations.csv."""
    out = {}
    diag = run_dir / "diagnostics"
    if not diag.is_dir():
        return out
    for node_dir in sorted(diag.iterdir()):
        f = node_dir / "initialisations.csv"
        if not f.is_file():
            continue
        df = pd.read_csv(f).set_index("init").sort_index()
        out[node_dir.name] = df
    return out


def load_results(run_dir: Path) -> dict:
    f = run_dir / "results.npy"
    if not f.is_file():
        return {}
    raw = np.load(f, allow_pickle=True).item()
    return {str(k): v for k, v in raw.items()}


def compare_inits(a: dict, b: dict, name_a: str, name_b: str) -> bool:
    ok = True
    shared = sorted(set(a) & set(b))
    only_a, only_b = sorted(set(a) - set(b)), sorted(set(b) - set(a))

    if only_a or only_b:
        ok = False
        print(f"  node sets differ: only in {name_a}: {only_a}; only in {name_b}: {only_b}")
    if not shared:
        print("  FAIL: no architectures in common. Did both runs use use_diagnostics: true?")
        return False

    print(f"\n  {'node':>14} {'inits':>6} {'max|dBIC|':>12} {'max|dAIC|':>12}  verdict")
    for node in shared:
        da, db = a[node], b[node]
        idx = da.index.intersection(db.index)
        if len(idx) == 0:
            print(f"  {node:>14} {'0':>6} {'-':>12} {'-':>12}  FAIL (no shared inits)")
            ok = False
            continue
        dbic = float(np.max(np.abs(da.loc[idx, "BIC"].to_numpy() - db.loc[idx, "BIC"].to_numpy())))
        daic = float(np.max(np.abs(da.loc[idx, "AIC"].to_numpy() - db.loc[idx, "AIC"].to_numpy())))
        good = (dbic == 0.0) and (daic == 0.0)
        ok &= good
        if len(idx) != len(da.index) or len(idx) != len(db.index):
            print(f"    note: {node} has {len(da.index)} vs {len(db.index)} inits, comparing {len(idx)}")
        print(f"  {node:>14} {len(idx):>6} {dbic:>12.6g} {daic:>12.6g}  {'PASS' if good else 'FAIL'}")
    return ok


def compare_results(a: dict, b: dict, name_a: str, name_b: str, label: str) -> bool:
    ok = True
    shared = sorted(set(a) & set(b))
    if not shared:
        print(f"\n  {label}: no architectures in common, skipped")
        return True
    print(f"\n  {label}: selected values, {name_a} vs {name_b}")
    print(f"  {'node':>14} {'dBIC':>14} {'dAIC':>14}  verdict")
    for node in shared:
        va, vb = a[node], b[node]
        if va is None or vb is None:
            print(f"  {node:>14} {'-':>14} {'-':>14}  SKIP (null)")
            continue
        dbic = abs(float(va[0]) - float(vb[0]))
        daic = abs(float(va[1]) - float(vb[1]))
        good = (dbic == 0.0) and (daic == 0.0)
        ok &= good
        print(f"  {node:>14} {dbic:>14.6g} {daic:>14.6g}  {'PASS' if good else 'FAIL'}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--serial", required=True, help="run dir estimated with parallel_over_inits: false")
    ap.add_argument("--parallel", required=True, help="run dir estimated with parallel_over_inits: true")
    ap.add_argument("--reference", default=None,
                    help="optional pre-refactor run dir, compared on results.npy only")
    a = ap.parse_args()

    serial, parallel = _resolve(a.serial), _resolve(a.parallel)
    for p in (serial, parallel):
        if not p.is_dir():
            print(f"missing run dir: {p}")
            return 1

    print("=" * 72)
    print("(node, init) parallelisation regression")
    print("=" * 72)
    print(f"  serial   : {serial.name}")
    print(f"  parallel : {parallel.name}")

    si, pi = load_inits(serial), load_inits(parallel)
    if not si or not pi:
        print("\n  FAIL: one or both runs have no diagnostics/<node>/initialisations.csv.")
        print("  Both runs must be estimated with use_diagnostics: true.")
        return 1

    print("\n-- per-initialisation agreement (the seeding question) --")
    ok = compare_inits(si, pi, "serial", "parallel")

    sr, pr = load_results(serial), load_results(parallel)
    ok &= compare_results(sr, pr, "serial", "parallel", "results.npy")

    if a.reference:
        ref = _resolve(a.reference)
        if ref.is_dir():
            print(f"\n-- against pre-refactor reference: {ref.name} --")
            rr = load_results(ref)
            ok &= compare_results(sr, rr, "serial", "reference", "results.npy")
        else:
            print(f"\n  reference run dir not found, skipped: {ref}")

    print("\n" + "=" * 72)
    if ok:
        print("PASS. Per-initialisation results are bit-identical across process layouts.")
        print("The (node, init) parallelisation is safe to dispatch.")
    else:
        print("FAIL. Do NOT dispatch the pilot.")
        print("A non-zero difference means initialisation j depends on process history,")
        print("so the parallel path is not the same estimator as the serial one.")
    print("=" * 72)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
