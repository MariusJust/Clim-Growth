"""Country-cluster bootstrap of the Burke and Leirvik benchmarks.

Same resampling design as the network bootstrap: countries drawn with
replacement, duplicates relabelled so each becomes a distinct unit with its own
fixed effect and trends. Results are cached so the projection script can attach
95% bands to the benchmark arms on equal footing with the network.

Optimisation vs the reference within_fit: the annihilator is built ONCE per draw
and applied to Burke's and Leirvik's regressors together, and the explicit
B = pinv(W'W) @ W.T (a 648 x 10000 product) is never formed -- we solve against
W.T @ V instead. That is roughly a 4x saving over refitting twice per draw.

    python scripts/2026-08-30-quad-bootstrap.py --n 1000

Writes/extends data/bootstrap_quad_coefs.npz (resumable).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
CACHE = ROOT / "data/bootstrap_quad_coefs.npz"

QB = ["T", "T2", "Pm", "Pm2"]
QL = ["T", "T2", "Pm", "Pm2", "TP", "T2P", "TP2", "T2P2"]


def load_panel():
    d = pd.read_csv(ROOT / "data/MainData_est.csv", low_memory=False)
    d = d.rename(columns={"GrowthWDI": "growth"})
    d["T"] = pd.to_numeric(d.TempPopWeight, errors="coerce")
    d["Pmm"] = pd.to_numeric(d.PrecipPopWeight, errors="coerce")
    d["growth"] = pd.to_numeric(d.growth, errors="coerce")
    d["iso"] = d.CountryCode.replace({"COD": "ZAR", "ROU": "ROM"})
    e = d.dropna(subset=["growth", "T", "Pmm"]).copy()
    e["Pm"] = e.Pmm / 1000.0
    e["T2"] = e["T"] ** 2; e["Pm2"] = e.Pm ** 2
    e["TP"] = e["T"] * e.Pm; e["T2P"] = e.T2 * e.Pm
    e["TP2"] = e["T"] * e.Pm2; e["T2P2"] = e.T2 * e.Pm2
    return e


def fit_both(unit, year, V):
    """unit/year: integer codes. V: [X_leirvik | y] stacked columns.

    Returns (beta_burke, beta_leirvik) using the same FWL annihilator:
    country FE + year FE + country linear and quadratic trends.
    """
    n = len(unit)
    nu, ny = unit.max() + 1, year.max() + 1
    t = year.astype(float); t -= t.mean()
    W = np.zeros((n, 2 * nu + ny))
    r = np.arange(n)
    W[r, unit] = 1.0                       # country FE
    W[r, nu + year] = 1.0                  # year FE
    W[r, nu + ny + unit] = t               # country linear trend
    W2 = np.zeros((n, nu))
    W2[r, unit] = t ** 2                   # country quadratic trend
    W = np.hstack([W, W2])

    WtW = W.T @ W
    G = W.T @ V
    c = np.linalg.pinv(WtW) @ G
    A = V - W @ c                          # annihilated [X | y]

    yv = A[:, -1]
    XL = A[:, :-1]
    XB = XL[:, :4]
    bB = np.linalg.lstsq(XB, yv, rcond=None)[0]
    bL = np.linalg.lstsq(XL, yv, rcond=None)[0]
    return bB, bL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=float, default=150.0,
                    help="seconds to work before saving and exiting (resumable)")
    a = ap.parse_args()

    e = load_panel()
    iso = np.array(sorted(e.iso.unique()))
    n_iso = len(iso)
    blocks = {c: sub for c, sub in e.groupby("iso")}
    colsL = QL + ["growth"]
    arrs = {c: blocks[c][colsL].to_numpy(float) for c in iso}
    yrs = {c: blocks[c]["Year"].to_numpy() for c in iso}
    year_codes = {y: i for i, y in enumerate(sorted(e.Year.unique()))}

    if CACHE.exists():
        z = np.load(CACHE)
        cB, cL = list(z["burke"]), list(z["leirvik"])
    else:
        cB, cL = [], []
    done = len(cB)
    print(f"panel {len(e)} rows, {n_iso} countries | cached draws: {done}/{a.n}")
    if done >= a.n:
        print("already complete"); return

    rng = np.random.default_rng(a.seed)
    for _ in range(done):                  # replay to keep the stream identical
        rng.integers(0, n_iso, size=n_iso)

    t0 = time.time()
    while len(cB) < a.n and time.time() - t0 < a.budget:
        draw = rng.integers(0, n_iso, size=n_iso)
        Vs, us, ys = [], [], []
        for i, j in enumerate(draw):
            c = iso[j]
            Vs.append(arrs[c])
            us.append(np.full(len(arrs[c]), i))
            ys.append([year_codes[y] for y in yrs[c]])
        V = np.vstack(Vs)
        unit = np.concatenate(us)
        year = np.concatenate([np.asarray(x) for x in ys])
        try:
            bB, bL = fit_both(unit, year, V)
            cB.append(bB); cL.append(bL)
        except Exception as exc:
            print("  draw failed:", exc)
            cB.append(np.full(4, np.nan)); cL.append(np.full(8, np.nan))
        if len(cB) % 100 == 0:
            el = time.time() - t0
            print(f"  {len(cB)}/{a.n} draws ({el:.0f}s, {el/(len(cB)-done):.2f}s per draw)")

    np.savez_compressed(CACHE, burke=np.array(cB), leirvik=np.array(cL))
    print(f"saved {len(cB)}/{a.n} draws -> {CACHE}")
    if len(cB) < a.n:
        print("budget reached; rerun the same command to continue")


if __name__ == "__main__":
    main()
