"""Time-varying climate response surfaces from the dynamic network.

Source: runs/estimation/2026-07-02_12-01-08_global_dynamic, BIC-best (2, 2).

    !! CAVEAT !! This run is dated 2 July 2026 and therefore predates the
    27 August 2026 correction to the population-weighted climate data. Its
    surfaces are NOT comparable in level with the corrected-data figures
    elsewhere in the paper. It is the only run with a time input, so it is the
    only source of a time-varying surface until a dynamic run is repeated on
    the corrected data.

Two implementation details that are easy to get wrong:

  * Time enters the network RAW, as ``year - (first_year - 1)`` (see the
    ``'time'`` branch of helper_functions/shared/vectorize.py). Only temperature
    and precipitation are standardised. ``create_pred_input`` builds
    ``arange(0, T+1)``, which starts at 0 and would shift the whole time axis by
    one period, so it is deliberately not used here.
  * Standardisation moments must come from the PRE-BUGFIX data the run was
    trained on, not from the corrected MainData.xlsx.

The panel is 1961-2023, so 1960 (t = 0) lies outside the trained range; the
first panel is 1961 (t = 1) and is labelled as such.

Because the model carries additive time fixed effects, the LEVEL of the surface
at any given year is not separately identified from that year's fixed effect.
Each year's surface is therefore centred at its own mean-climate cell, and only
the SHAPE is interpretable across years.

Outputs: paper/Figures/Surfaces/time_varying_surfaces.{pdf,png}
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hdf5_min import read_weights                                   # noqa: E402

RUN = ROOT / "runs/estimation/2026-07-02_12-01-08_global_dynamic"
WEIGHTS = RUN / "parameters/(2, 2).weights.h5"
PREFIX_DATA = ROOT / "data/archive/2026-08-27-MainData-PREBUGFIX.xlsx"
MOMENTS = ROOT / "data/prebugfix_moments.npy"
OUT = ROOT / "paper/Figures/Surfaces"
OUT.mkdir(parents=True, exist_ok=True)

YEARS = [1961, 1980, 2000, 2010, 2020]
DATA_END = 2024


def swish(z):
    return z / (1.0 + np.exp(-z))


def moments():
    """Standardisation from the data the dynamic run was TRAINED on."""
    if MOMENTS.exists():
        return np.load(MOMENTS)
    raw = pd.read_excel(PREFIX_DATA,
                        usecols=["Year", "CountryCode", "TempPopWeight",
                                 "PrecipPopWeight"])
    raw = raw[raw["Year"] <= DATA_END]
    pT = raw.pivot_table(index="Year", columns="CountryCode",
                         values="TempPopWeight")
    pP = raw.pivot_table(index="Year", columns="CountryCode",
                         values="PrecipPopWeight")
    m = np.array([np.nanmean(pT.values), np.nanstd(pT.values),
                  np.nanmean(pP.values), np.nanstd(pP.values)])
    np.save(MOMENTS, m)
    return m


def main():
    mT, sT, mP, sP = moments()
    print(f"pre-bugfix moments: T {mT:.3f} +- {sT:.3f} | P {mP:.1f} +- {sP:.1f}")

    w = read_weights(str(WEIGHTS))
    K1 = np.asarray(w["layers/dense/vars/0"])      # (3, 2)  inputs T, P, time
    b1 = np.asarray(w["layers/dense/vars/1"])
    K2 = np.asarray(w["layers/dense_1/vars/0"])    # (2, 2)
    b2 = np.asarray(w["layers/dense_1/vars/1"])
    Ko = np.asarray(w["layers/dense_2/vars/0"]).ravel()   # (2,) no output bias

    # first year of the training panel -> time index mapping
    d = pd.read_csv(ROOT / "data/MainData_est.csv", low_memory=False)
    d["T"] = pd.to_numeric(d.TempPopWeight, errors="coerce")
    yrs = sorted(d[d.Year <= DATA_END].dropna(subset=["T"]).Year.unique())
    y0 = int(yrs[0])
    print(f"training panel {y0}..{int(yrs[-1])}  ->  t = year - {y0 - 1}")

    Tv = np.linspace(0, 30, 90)
    Pv = np.linspace(12.03731002, 3000.0, 90)
    T, P = np.meshgrid(Tv, Pv)
    Tz, Pz = (T - mT) / sT, (P - mP) / sP

    def surface(t_index):
        X = np.stack([Tz.ravel(), Pz.ravel(),
                      np.full(Tz.size, float(t_index))], axis=-1)   # (N, 3)
        h1 = swish(X @ K1 + b1)
        h2 = swish(h1 @ K2 + b2)
        return (h2 @ Ko).reshape(T.shape)

    surfs, labels = [], []
    ci, cj = np.argmin(np.abs(Pv - mP)), np.argmin(np.abs(Tv - mT))
    for y in YEARS:
        t = y - (y0 - 1)
        s = surface(t)
        surfs.append(s - s[ci, cj])          # centre: level is absorbed by time FE
        labels.append(f"{y}  (t={t})")
        print(f"  {y}: t={t:3d}  optimum at mean precip = "
              f"{Tv[np.argmax(surfs[-1][ci, :])]:.2f} C")

    A = np.array(surfs) * 100.0
    vmax = np.nanpercentile(np.abs(A), 99)
    lv = np.linspace(-vmax, vmax, 21)

    fig = plt.figure(figsize=(15.5, 6.8))
    gs = fig.add_gridspec(2, len(YEARS), height_ratios=[1.25, 1.0], hspace=0.42)
    for i, (s, lab) in enumerate(zip(A, labels)):
        ax = fig.add_subplot(gs[0, i])
        cf = ax.contourf(T, P / 1000.0, s, levels=lv, cmap="RdBu_r",
                         vmin=-vmax, vmax=vmax, extend="both")
        ax.contour(T, P / 1000.0, s, levels=[0], colors="k", linewidths=0.6)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("temperature ($\\degree$C)", fontsize=8.5)
        if i == 0:
            ax.set_ylabel("precipitation (m)", fontsize=8.5)
        ax.tick_params(labelsize=7.5)
    cax = fig.add_axes([0.92, 0.56, 0.011, 0.30])
    fig.colorbar(cf, cax=cax).set_label("growth effect (pp)", fontsize=8.5)

    ax = fig.add_subplot(gs[1, :])
    # Reference: the STATIC corrected-data response on the same axes. The
    # dynamic model spans only 1-5 pp across temperature against ~19 pp here,
    # because with no additive time FE it has spent its capacity on the time
    # trend and saturated its temperature-carrying unit off (see printout).
    try:
        wS = read_weights(str(ROOT / "runs/estimation/"
                              "2026-08-27_11-01-35_global_IC_country_trends"
                              "/parameters/(2,).weights.h5"))
        KS = np.asarray(wS["layers/dense/vars/0"])
        cS = np.asarray(wS["layers/dense/vars/1"])
        bS = np.asarray(wS["layers/dense_1/vars/0"]).ravel()
        mT2, sT2, mP2, sP2 = 20.3801, 7.2308, 1281.5723, 833.7378
        zc = ((Tv - mT2) / sT2)[:, None] * KS[0] + (0.0) * KS[1] + cS
        stat = 100.0 * (swish(zc) @ bS)
        stat = stat - stat[np.argmin(np.abs(Tv - mT2))]
        ax.plot(Tv, stat, color="#999999", lw=1.6, ls="--",
                label="static (2,), corrected data — reference")
    except Exception as exc:
        print("  static reference unavailable:", exc)

    cols = plt.cm.viridis(np.linspace(0, 0.9, len(YEARS)))
    for s, lab, c in zip(A, labels, cols):
        rng = s[ci, :].max() - s[ci, :].min()
        ax.plot(Tv, s[ci, :], color=c, lw=2.0,
                label=f"{lab.split('  ')[0]} (range {rng:.1f} pp)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("annual mean temperature ($\\degree$C)")
    ax.set_ylabel("growth effect (pp)")
    ax.set_title("cross-section at mean precipitation, each year centred at its "
                 "own mean-climate cell", fontsize=9.5, loc="left")
    ax.set_xlim(0, 30)
    ax.legend(frameon=False, fontsize=8.5, ncol=len(YEARS))
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25, lw=0.6)

    fig.suptitle("Time-varying climate response surface, dynamic network (2, 2)  "
                 "— estimated on PRE-BUGFIX climate data (run of 2 July 2026)",
                 fontsize=11, x=0.01, ha="left")
    fig.subplots_adjust(left=0.05, right=0.90, top=0.90, bottom=0.08)
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"time_varying_surfaces.{e}", dpi=170,
                    bbox_inches="tight")
    print(f"wrote {OUT/'time_varying_surfaces.pdf'}")


if __name__ == "__main__":
    main()
