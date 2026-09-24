"""Pre-flight checks for the within projection before submitting an EE job.

Run this on the server (where TensorFlow exists) BEFORE `python scripts/slurm.py`.
It is cheap and it fails loudly, so it is a much better way to find a wiring
mistake than watching a 9-hour job die at architecture 3.

    PYTHONPATH=src python scripts/test_within_projector.py            # default
    PYTHONPATH=src python scripts/test_within_projector.py --full     # + full ee panel (~4 GB)
    PYTHONPATH=src python scripts/test_within_projector.py --no-tf    # skip TF checks

Checks, in order of what they would catch:

  1 REGRESSION   the unweighted, ungrouped projector must be numerically
                 identical to the original implementation, because every wb
                 result already in the paper depends on it.
  2 WEIGHTS      the weighted problem must actually solve WᵀΩWγ = WᵀΩy, and
                 weights=1 must reproduce the unweighted operator exactly.
  3 GROUPING     trend_groups must shrink p, and trend_groups=arange(N) must
                 reproduce per-unit trends bit-for-bit.
  4 ROW ORDER    the projector indexes with argwhere(~mask) (time-major) while
                 the Keras loss uses boolean_mask(reshape(...)). If those two
                 ever disagree, every fitted value is silently permuted. This is
                 the classic way a subnational port breaks.
  5 TF LOSS      the Keras loss, built exactly as multivariate_model builds it,
                 must equal the numpy annihilator on the same inputs.
  6 EE PANEL     the real admin-1 design builds, with the expected p, rank and
                 Σw, and the memory/time it costs per worker.

Exit code is 0 only if every check passes.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

# Load the module by path: importing it through the package would pull in
# models/__init__ -> tensorflow, and checks 1-4 are pure numpy and must run
# even where TF is absent (e.g. a laptop or a login node without the env).
import importlib.util as _ilu                                            # noqa: E402

_spec = _ilu.spec_from_file_location(
    "_within_projection",
    ROOT / "src/models/helper_functions/global_model/within_projection.py")
_wp = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_wp)
WithinProjector = _wp.WithinProjector

FAILS = []
TOL = 1e-9


def check(name, value, tol=TOL, note=""):
    ok = bool(np.all(np.abs(value) <= tol))
    print(f"    [{'PASS' if ok else 'FAIL'}] {name:52s} {value:.2e}"
          + (f"   {note}" if note else ""))
    if not ok:
        FAILS.append(name)
    return ok


def check_true(name, ok, note=""):
    print(f"    [{'PASS' if ok else 'FAIL'}] {name:52s} " + (note or ""))
    if not ok:
        FAILS.append(name)
    return ok


def reference_B(W):
    """The original implementation's dense annihilator factor."""
    return np.linalg.pinv(W.T @ W) @ W.T


def small_mask(T=20, N=30, p_missing=0.10, seed=1):
    return np.random.default_rng(seed).random((T, N)) < p_missing


# --------------------------------------------------------------------------- #
def test_regression():
    print("\n  1 REGRESSION - unweighted path must match the original code")
    mask = small_mask()
    rng = np.random.default_rng(2)
    for include_time, lab in ((True, "static"), (False, "dynamic (no time FE)")):
        p = WithinProjector(mask, include_time=include_time)
        B = reference_B(p.W)
        v = rng.normal(size=p.n_obs)
        check(f"{lab}: annihilate == v - W(WᵀW)⁺Wᵀv", np.abs((v - p.W @ (B @ v)) - p.annihilate(v)).max())
        check(f"{lab}: recover_gamma == (WᵀW)⁺Wᵀr", np.abs((B @ v) - p.recover_gamma(v)).max())
        check(f"{lab}: legacy .B property", np.abs(p.B - B).max())
        check_true(f"{lab}: rank == matrix_rank(W)",
                   p.rank == int(np.linalg.matrix_rank(p.W)),
                   f"rank={p.rank}, p={p.p}")

    # the design matrix itself must be unchanged, column for column
    obs = ~mask
    idx = np.argwhere(obs)
    t_a, n_a = idx[:, 0], idx[:, 1]
    m, (T, N) = len(idx), mask.shape
    Wc = np.zeros((m, N)); Wc[np.arange(m), n_a] = 1.0
    Wt = np.zeros((m, T)); Wt[np.arange(m), t_a] = 1.0
    tc = t_a.astype(float); tc -= tc.mean()
    W_old = np.concatenate([Wc, Wt, Wc * tc[:, None], Wc * (tc ** 2)[:, None]], axis=1)
    p = WithinProjector(mask)
    check_true("design matrix shape unchanged", p.W.shape == W_old.shape,
               f"{p.W.shape}")
    check("design matrix identical to old construction", np.abs(p.W - W_old).max(), tol=0.0)


def test_weights():
    print("\n  2 WEIGHTS - the weighted normal equations must actually be solved")
    mask = small_mask()
    rng = np.random.default_rng(3)
    N = mask.shape[1]
    w_unit = rng.uniform(0.3, 2.0, size=N)
    p = WithinProjector(mask, weights=w_unit)
    y = rng.normal(size=p.n_obs)
    g = p.recover_gamma(y)
    lhs = p.W.T @ (p.w[:, None] * p.W) @ g
    rhs = p.W.T @ (p.w * y)
    check("WᵀΩWγ == WᵀΩy", np.abs(lhs - rhs).max(), tol=1e-8)
    r = p.annihilate(y)
    check("weighted residual ⟂ weighted design", np.abs(p.W.T @ (p.w * (r / p.s))).max(), tol=1e-8)
    check_true("Σw recorded", np.isclose(p.sum_w, w_unit[p.n_arr].sum()),
               f"sum_w={p.sum_w:.3f}")
    p0 = WithinProjector(mask)
    p1 = WithinProjector(mask, weights=np.ones(N))
    v = rng.normal(size=p0.n_obs)
    check("weights=1 identical to unweighted", np.abs(p1.annihilate(v) - p0.annihilate(v)).max(), tol=0.0)


def test_grouping():
    print("\n  3 GROUPING - per-country trends")
    mask = small_mask()
    N = mask.shape[1]
    rng = np.random.default_rng(4)
    p_unit = WithinProjector(mask)
    p_grp = WithinProjector(mask, trend_groups=np.repeat(np.arange(6), N // 6))
    p_id = WithinProjector(mask, trend_groups=np.arange(N))
    check_true("grouping reduces p", p_grp.p < p_unit.p, f"{p_unit.p} -> {p_grp.p}")
    v = rng.normal(size=p_unit.n_obs)
    check("trend_groups=arange(N) == per-unit trends",
          np.abs(p_id.annihilate(v) - p_unit.annihilate(v)).max(), tol=0.0)
    try:
        WithinProjector(mask, trend_groups=np.arange(N + 3))
        check_true("wrong-length trend_groups rejected", False)
    except ValueError:
        check_true("wrong-length trend_groups rejected", True)
    try:
        WithinProjector(mask, weights=np.zeros(N))
        check_true("non-positive weights rejected", False)
    except ValueError:
        check_true("non-positive weights rejected", True)


def test_row_order():
    """The projector's argwhere(~mask) order must equal the loss's flatten order."""
    print("\n  4 ROW ORDER - projector indexing vs the Keras loss's boolean_mask")
    mask = small_mask()
    T, N = mask.shape
    p = WithinProjector(mask)
    y = np.arange(T * N, dtype=float).reshape(T, N)          # every cell distinct
    from_proj = y[p.t_arr, p.n_arr]                          # projector's order
    from_loss = y.reshape(-1)[~mask.reshape(-1)]             # Keras loss's order
    check("argwhere(~mask) order == reshape/boolean_mask order",
          np.abs(from_proj - from_loss).max(), tol=0.0,
          note=f"n_obs={p.n_obs}")


def test_tf_loss():
    print("\n  5 TF LOSS - Keras loss must equal the numpy annihilator")
    try:
        import tensorflow as tf
    except Exception as exc:                                  # noqa: BLE001
        print(f"    [SKIP] tensorflow unavailable ({type(exc).__name__})")
        return
    from models.helper_functions.global_model.loss import individual_loss

    mask = small_mask(T=12, N=15)
    T, N = mask.shape
    rng = np.random.default_rng(5)
    for w_unit, lab in ((None, "unweighted"), (rng.uniform(0.3, 2.0, N), "weighted")):
        p = WithinProjector(mask, weights=w_unit)
        y = rng.normal(size=(T, N))
        f = rng.normal(size=(T, N))
        y_obs = y[p.t_arr, p.n_arr]
        within = dict(
            W=tf.constant(p.W, dtype=tf.float32),
            M=tf.constant(p.M, dtype=tf.float32),
            s=tf.constant(p.s, dtype=tf.float32),
            w=tf.constant(p.w, dtype=tf.float32),
            Py=tf.constant(p.annihilate(y_obs), dtype=tf.float32),
        )
        loss_fn = individual_loss(mask=tf.constant(mask), within=within)
        got = float(loss_fn(tf.constant(y[None, :, :], dtype=tf.float32),
                            tf.constant(f[None, :, :], dtype=tf.float32)))
        want = float(np.mean(p.annihilate(y_obs - f[p.t_arr, p.n_arr]) ** 2))
        check(f"{lab}: keras loss == mean(P(y-f)²)", abs(got - want) / max(want, 1e-12),
              tol=2e-5, note=f"loss={got:.6f}")


def test_ee_panel(full=False):
    print("\n  6 EE PANEL - the real admin-1 design")
    import pandas as pd
    src = ROOT / "data/ee_data.csv"
    if not src.exists():
        print(f"    [SKIP] {src} not found")
        return
    d = pd.read_csv(src, sep=";", usecols=["fid", "iso3", "Year", "growth (gdp per capita)"],
                    low_memory=False).rename(columns={"growth (gdp per capita)": "g"})
    fid2iso = d.dropna(subset=["fid", "iso3"]).drop_duplicates("fid").set_index("fid")["iso3"]
    if not full:
        keep = np.random.default_rng(0).choice(sorted(fid2iso.index), 700, replace=False)
        d = d[d.fid.isin(keep)]
        print("    (subsampled to 700 units; pass --full for the whole panel, ~4 GB)")
    piv = d.pivot_table(index="Year", columns="fid", values="g").iloc[1:, :]
    mask = piv.isna().to_numpy()
    iso = np.array([fid2iso[int(c)] for c in piv.columns])
    per = dict(zip(*np.unique(iso, return_counts=True)))
    w = np.array([1.0 / per[c] for c in iso])
    T, N = mask.shape
    nC = len(per)

    t0 = time.time()
    p = WithinProjector(mask, country_trends=True, quadratic_trends=True,
                        include_time=True, trend_groups=iso, weights=w)
    el = time.time() - t0
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2
    except Exception:                                          # noqa: BLE001
        rss = float("nan")

    print(f"    panel {T} years x {N} units, {nC} countries, n_obs={p.n_obs}")
    check_true("p == N + T + 2*n_countries", p.p == N + T + 2 * nC,
               f"p={p.p} (expected {N + T + 2 * nC})")
    check_true("Σw == n_countries x n_years", np.isclose(p.sum_w, nC * T),
               f"Σw={p.sum_w:.0f} (expected {nC * T})")
    check_true("rank <= p and rank > 0", 0 < p.rank <= p.p, f"rank={p.rank}")
    v = np.random.default_rng(0).normal(size=p.n_obs)
    r = p.annihilate(v)
    check("weighted residual ⟂ design", np.abs(p.W.T @ (p.w * (r / p.s))).max(), tol=1e-7)
    print(f"    build {el:.1f}s   peak RSS {rss:.2f} GB   "
          f"W {p.W.nbytes / 1024 ** 3:.2f} GB + M {p.M.nbytes / 1024 ** 3:.3f} GB")
    if full:
        per_worker = (p.W.nbytes * 1.5 + p.M.nbytes) / 1024 ** 3
        print(f"    -> ~{per_worker:.2f} GB/worker incl. the float32 TF copy; "
              f"x55 = {55 * per_worker:.0f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="build the whole ee panel (~4 GB)")
    ap.add_argument("--no-tf", action="store_true", help="skip the TensorFlow check")
    a = ap.parse_args()

    print("=" * 78)
    print("  within-projection pre-flight")
    print("=" * 78)
    test_regression()
    test_weights()
    test_grouping()
    test_row_order()
    if not a.no_tf:
        test_tf_loss()
    test_ee_panel(full=a.full)

    print("\n" + "=" * 78)
    if FAILS:
        print(f"  {len(FAILS)} CHECK(S) FAILED - do not submit:")
        for f in FAILS:
            print(f"    - {f}")
        return 1
    print("  all checks passed")
    print("  NOTE: this does not exercise a full training step. Watch the first")
    print("        architecture complete before letting all 17 run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
