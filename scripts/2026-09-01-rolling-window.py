"""Rolling-window estimates of the climate-growth response: is it time-varying?

Fits the Burke quadratic (country + year FE, country linear and quadratic trends)
on 20-year rolling windows and plots two well-behaved summaries with
country-cluster bootstrap CIs:

  Panel A  b_T2, the curvature. Negative = concave = an interior optimum exists.
  Panel B  the marginal effect of +1 C at 25 C, b_T + 2*b_T2*25, which is the
           economically interesting object for hot countries.

We deliberately do NOT plot T* = -b_T/(2*b_T2): it is a ratio that explodes as the
curvature approaches zero, which is exactly what happens in the late windows. The
apparent "optimum shifts from 12 C to 29 C" in a linear time-interaction model is
that artefact, not a shift of a stable optimum.

Two country sets are shown because composition turns out to matter:
  all countries  - the panel grows from 143 to 191 countries across windows
  fixed set      - the 87 countries present in every year 1961-2023

Outputs: paper/Figures/TimeVarying/rolling_window.{pdf,png,csv}
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper/Figures/TimeVarying"
OUT.mkdir(parents=True, exist_ok=True)

X = ["T", "T2", "Pm", "Pm2"]
WIN = 20
N_BOOT = 200
EVAL_T = 25.0


def load():
    b = pd.read_csv(ROOT / "data/MainData_est.csv", low_memory=False)
    b = b.rename(columns={"GrowthWDI": "growth"})
    for c in ["TempPopWeight", "PrecipPopWeight", "growth"]:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b["T"] = b.TempPopWeight
    b["Pm"] = b.PrecipPopWeight / 1000.0
    b = b.dropna(subset=["growth", "T", "Pm"]).copy()
    b["iso"] = b.CountryCode.replace({"COD": "ZAR", "ROU": "ROM"})
    b["T2"] = b["T"] ** 2
    b["Pm2"] = b.Pm ** 2
    return b


def fit(V, unit, year):
    """FWL: country + year FE, country linear and quadratic trends."""
    n = len(unit)
    nu, ny = int(unit.max()) + 1, int(year.max()) + 1
    t = year.astype(float); t -= t.mean()
    r = np.arange(n)
    D1 = np.zeros((n, nu)); D1[r, unit] = 1.0
    D2 = np.zeros((n, ny)); D2[r, year] = 1.0
    L = np.zeros((n, nu)); L[r, unit] = t
    Q = np.zeros((n, nu)); Q[r, unit] = t ** 2
    W = np.hstack([D1, D2, L, Q])
    A = V - W @ (np.linalg.pinv(W.T @ W) @ (W.T @ V))
    return np.linalg.lstsq(A[:, :-1], A[:, -1], rcond=None)[0]


def window_stats(e, rng, n_boot):
    """Point estimate + bootstrap draws of (b_T2, marginal effect at EVAL_T)."""
    cols = X + ["growth"]
    V = e[cols].to_numpy(float)
    unit = pd.Categorical(e.iso).codes.astype(np.int64)
    year = pd.Categorical(e.Year).codes.astype(np.int64)
    b = fit(V, unit, year)
    pt = (b[1], b[0] + 2 * b[1] * EVAL_T)

    iso = e.iso.to_numpy()
    uniq = np.unique(iso)
    idx = {c: np.flatnonzero(iso == c) for c in uniq}
    yr = e.Year.to_numpy()
    draws = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), size=len(uniq))
        rows, un = [], []
        for k, j in enumerate(pick):
            ii = idx[uniq[j]]
            rows.append(ii); un.append(np.full(len(ii), k))
        rows = np.concatenate(rows); un = np.concatenate(un)
        yy = pd.Categorical(yr[rows]).codes.astype(np.int64)
        try:
            bb = fit(V[rows], un, yy)
            draws.append((bb[1], bb[0] + 2 * bb[1] * EVAL_T))
        except Exception:
            pass
    return pt, np.array(draws)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default=None)
    a = ap.parse_args()
    CACHE = ROOT / "data/rolling_window_cache"
    CACHE.mkdir(exist_ok=True)
    b = load()
    full = b.groupby("iso").Year.nunique()
    keep = full[full == b.Year.nunique()].index
    sets = {"all countries": b, f"fixed set ({len(keep)} countries)": b[b.iso.isin(keep)]}
    starts = list(range(1961, 2005, 4))

    res, t0 = {}, time.time()
    for lab, dat in sets.items():
        tag = "all" if lab.startswith("all") else "fixed"
        cf = CACHE / f"{tag}.csv"
        if a.only and a.only != tag:
            if cf.exists():
                res[lab] = pd.read_csv(cf); print(f"  [cached] {lab}")
                continue
            else:
                print(f"  [skip] {lab} (no cache yet)"); continue
        if cf.exists() and not a.only:
            res[lab] = pd.read_csv(cf); print(f"  [cached] {lab}"); continue
        # window-level checkpointing so a long run can resume across invocations
        done = pd.read_csv(cf) if cf.exists() else pd.DataFrame(columns=["start"])
        rows = done.to_dict("records")
        have = set(done["start"].tolist()) if len(done) else set()
        rng = np.random.default_rng(0)
        for s in starts:
            if s in have:
                continue
            if time.time() - t0 > 140:
                print("  [budget reached - rerun to continue]"); break
            e = dat[(dat.Year >= s) & (dat.Year <= s + WIN - 1)]
            if e.Year.nunique() < 15 or e.iso.nunique() < 30:
                continue
            pt, dr = window_stats(e, rng, N_BOOT)
            lo = np.percentile(dr, 2.5, axis=0) if len(dr) else (np.nan, np.nan)
            hi = np.percentile(dr, 97.5, axis=0) if len(dr) else (np.nan, np.nan)
            rows.append(dict(mid=s + (WIN - 1) / 2, start=s, end=s + WIN - 1,
                             n=len(e), g=e.iso.nunique(),
                             bT2=pt[0], bT2_lo=lo[0], bT2_hi=hi[0],
                             me=pt[1], me_lo=lo[1], me_hi=hi[1]))
            print(f"  {lab:26s} {s}-{s+WIN-1}  b_T2={pt[0]*1000:+7.3f}e-3 "
                  f"[{lo[0]*1000:+6.3f},{hi[0]*1000:+6.3f}]  "
                  f"dg/dT@25={pt[1]*100:+6.2f}pp [{lo[1]*100:+6.2f},{hi[1]*100:+6.2f}]",
                  flush=True)
            pd.DataFrame(rows).sort_values("start").to_csv(cf, index=False)
        res[lab] = pd.DataFrame(rows).sort_values("start")
    print(f"  ({time.time()-t0:.0f}s)")

    res = {k: v for k, v in res.items() if len(v)}
    if len(res) < len(sets):
        print("  (not all samples available yet - rerun with --only for the rest)")
    C = {"all countries": "#b23a48"}
    for k in sets:
        C.setdefault(k, "#2f6f9f")
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.6))
    for lab, r in res.items():
        c = C[lab]
        ax[0].fill_between(r.mid, r.bT2_lo * 1000, r.bT2_hi * 1000, color=c, alpha=0.15, lw=0)
        ax[0].plot(r.mid, r.bT2 * 1000, color=c, lw=2, marker="o", ms=3.5, label=lab)
        ax[1].fill_between(r.mid, r.me_lo * 100, r.me_hi * 100, color=c, alpha=0.15, lw=0)
        ax[1].plot(r.mid, r.me * 100, color=c, lw=2, marker="o", ms=3.5, label=lab)
    ax[0].set_ylabel(r"$b_{T^2}\times 10^{3}$  (negative = concave)")
    ax[0].set_title("A. curvature of the temperature response", fontsize=10.5, loc="left")
    ax[1].set_ylabel("pp growth per +1 $\\degree$C")
    ax[1].set_title(f"B. marginal effect of warming at {EVAL_T:.0f} $\\degree$C",
                    fontsize=10.5, loc="left")
    for a in ax:
        a.axhline(0, color="grey", lw=0.8)
        a.set_xlabel(f"{WIN}-year window midpoint")
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.6)
        a.legend(frameon=False, fontsize=8.5)
    fig.suptitle(f"Rolling {WIN}-year windows, 95% country-cluster bootstrap "
                 f"({N_BOOT} draws per window)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    for e_ in ("pdf", "png"):
        fig.savefig(OUT / f"rolling_window.{e_}", dpi=170, bbox_inches="tight")
    pd.concat([r.assign(sample=k) for k, r in res.items()]).to_csv(
        OUT / "rolling_window.csv", index=False)
    print(f"wrote {OUT/'rolling_window.pdf'}")


if __name__ == "__main__":
    main()
