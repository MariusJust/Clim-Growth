#!/usr/bin/env python3
"""Learned factors of the BIC-selected global network.

The reported global model is a single hidden layer with two swish neurons:

    NN_k,it = c_k + gamma_k,T * T~_it + gamma_k,P * P~_it
    A_k,it  = swish(NN_k,it) = NN * sigmoid(NN)
    y_it    = beta_1 A_1,it + beta_2 A_2,it + mu_i + nu_t + trends + u_it

so A_it = (A_1,it, A_2,it)' can be read as two factors learned nonlinearly from
(T_it, P_it), with beta as the loadings. This script extracts (gamma, c, beta),
computes the factors for every observed country-year, and plots them.

Pure numpy/pandas/matplotlib - no TensorFlow required.
Run from the repo root:  python scripts/2026-08-25-learned-factors.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/sessions/charming-admiring-fermat/mnt/.claude/skills/paper1-analysis/scripts")
from hdf5_min import read_weights  # noqa: E402

WEIGHTS = ROOT / "runs/estimation/2026-06-13_22-11-06_global_IC_country_trends/parameters/(2,).weights.h5"
OUTDIR = ROOT / "paper/Figures/Interpretability"
OUTDIR.mkdir(parents=True, exist_ok=True)

DARKBLUE, BORDEAUX, GREY = "#26426b", "#b23a48", "#7a8698"


def swish(x):
    return x / (1.0 + np.exp(-x))


def main():
    # ---- parameters -------------------------------------------------------
    w = read_weights(str(WEIGHTS))
    G = np.asarray(w["layers/dense/vars/0"])          # (inputs=2, neurons=2)
    c = np.asarray(w["layers/dense/vars/1"])          # (2,)
    beta = np.asarray(w["layers/dense_1/vars/0"]).ravel()

    # ---- data + training standardisation ----------------------------------
    cols = ["CountryCode", "CountryName", "Year", "TempPopWeight", "PrecipPopWeight"]
    cache = ROOT / "data/MainData_slim.csv"          # xlsx read is slow; cache the columns
    if cache.exists():
        d = pd.read_csv(cache)
    else:
        d = pd.read_excel(ROOT / "data/MainData.xlsx", usecols=cols)
        d.to_csv(cache, index=False)
    d = d[cols].copy()
    d["T"] = pd.to_numeric(d["TempPopWeight"], errors="coerce")
    d["P"] = pd.to_numeric(d["PrecipPopWeight"], errors="coerce")   # mm, as in training
    d = d.dropna(subset=["T", "P"])
    mT, sT = np.nanmean(d["T"]), np.nanstd(d["T"])
    mP, sP = np.nanmean(d["P"]), np.nanstd(d["P"])
    Tz = (d["T"].to_numpy() - mT) / sT
    Pz = (d["P"].to_numpy() - mP) / sP

    # ---- factors ----------------------------------------------------------
    for k in (0, 1):
        d[f"NN{k+1}"] = c[k] + G[0, k] * Tz + G[1, k] * Pz
        d[f"A{k+1}"] = swish(d[f"NN{k+1}"].to_numpy())
    d["climate"] = beta[0] * d["A1"] + beta[1] * d["A2"]

    print(f"observations: {len(d):,}   countries: {d.CountryCode.nunique()}   "
          f"years: {int(d.Year.min())}-{int(d.Year.max())}")
    print(f"beta_1 = {beta[0]: .4f}   beta_2 = {beta[1]: .4f}")

    # ---- aggregates -------------------------------------------------------
    # IMPORTANT: aggregate over a BALANCED sub-panel. The full panel grows from
    # 103 countries (1961) to 194 (2010), with a jump of 18 countries between
    # 1990 and 1991 (Soviet breakup). Averaging over the unbalanced panel makes
    # mean temperature appear to FALL around 1991, which is pure composition.
    # On the balanced panel mean temperature rises by +1.13 degC over 1961-2023.
    nyr = d.Year.nunique()
    cover = d.groupby("CountryCode").Year.nunique()
    bal = set(cover[cover == nyr].index)
    db = d[d.CountryCode.isin(bal)]
    print(f"balanced sub-panel: {len(bal)} of {d.CountryCode.nunique()} countries "
          f"with full {nyr}-year coverage")

    byyr = db.groupby("Year").agg(A1=("A1", "mean"), A2=("A2", "mean"),
                                  T=("T", "mean"), P=("P", "mean"),
                                  climate=("climate", "mean")).reset_index()
    q = db.groupby("Year")[["A1", "A2"]].quantile([0.1, 0.5, 0.9]).unstack()

    print("\ncorrelation of cross-country means, 1960-2024:")
    for k in ("A1", "A2", "climate"):
        print(f"  corr({k}, T) = {byyr[k].corr(byyr['T']): .3f}   "
              f"corr({k}, P) = {byyr[k].corr(byyr['P']): .3f}")
    print(f"  corr(A1, A2)  = {byyr['A1'].corr(byyr['A2']): .3f}")

    # ---- figure -----------------------------------------------------------
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    # (a) mean factors
    a = ax[0, 0]
    a.plot(byyr.Year, byyr.A1, color=DARKBLUE, lw=1.8, label=r"$\bar A_{1t}$")
    a.plot(byyr.Year, byyr.A2, color=BORDEAUX, lw=1.8, label=r"$\bar A_{2t}$")
    a.set_title("(a)  Learned factors, balanced-panel mean", fontsize=10, loc="left")
    a.set_ylabel("factor value")
    a.legend(frameon=False, fontsize=9)

    # (b) mean climate inputs
    a = ax[0, 1]
    a.plot(byyr.Year, byyr["T"], color=DARKBLUE, lw=1.8, label=r"$\bar T_t$ ($^\circ$C)")
    a.set_ylabel(r"temperature ($^\circ$C)", color=DARKBLUE)
    a.tick_params(axis="y", labelcolor=DARKBLUE)
    a2 = a.twinx()
    a2.plot(byyr.Year, byyr["P"] / 1000.0, color=BORDEAUX, lw=1.8, ls="--",
            label=r"$\bar P_t$ (m)")
    a2.set_ylabel("precipitation (m)", color=BORDEAUX)
    a2.tick_params(axis="y", labelcolor=BORDEAUX)
    a.set_title("(b)  Climate inputs, balanced-panel mean", fontsize=10, loc="left")

    # (c)/(d) percentile bands
    for j, (k, col) in enumerate([("A1", DARKBLUE), ("A2", BORDEAUX)]):
        a = ax[1, j]
        lo, md, hi = q[(k, 0.1)], q[(k, 0.5)], q[(k, 0.9)]
        a.fill_between(lo.index, lo.values, hi.values, color=col, alpha=0.18,
                       lw=0, label="10th-90th pct")
        a.plot(md.index, md.values, color=col, lw=1.8, label="median")
        a.set_title(f"({'cd'[j]})  $A_{{{j+1}}}$ across countries (balanced)", fontsize=10, loc="left")
        a.set_xlabel("year")
        a.set_ylabel("factor value")
        a.legend(frameon=False, fontsize=9)

    for a in ax.ravel():
        a.spines[["top", "right"]].set_visible(False)
        a.grid(alpha=0.25, lw=0.6)
    ax[0, 1].spines["right"].set_visible(True)

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUTDIR / f"learned_factors.{ext}", dpi=160, bbox_inches="tight")
    print(f"\nwrote {OUTDIR/'learned_factors.pdf'}")

    d.to_csv(ROOT / "results/tables/learned_factors_panel.csv", index=False)
    print(f"wrote {ROOT/'results/tables/learned_factors_panel.csv'}")


if __name__ == "__main__":
    main()
