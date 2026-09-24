"""Time-varying climate response surface from the dynamic model.

Answers one question: is the estimated response surface time-varying, or is it
effectively static?

    python scripts/2026-09-08-dynamic-time-surface.py
    python scripts/2026-09-08-dynamic-time-surface.py --run <run_dir> --node "(8, 8, 4)"

Axes follow the request: temperature on x, YEARS ON Y, response on z, Cividis.

What is plotted
---------------
The surface is centred at the sample-mean temperature *within each year*:

    g(T, t) = f(T, Pmed, t) - mean_T f(T, Pmed, t)

That removes the common time level and leaves the shape of the temperature
response. The centring is not cosmetic. The dynamic model carries no additive
year fixed effect, so its t input is the only thing available to fit common year
effects, and it does so: the fitted common time component spans 16.10 pp against
the static model's year-FE spread of 14.75 pp. The level in t is therefore
confounded with the year effects and is not interpretable; the shape is what the
figure is about. See notes/2026-09-04/2026-09-04-dynamic-he-normal-run.md.

Also writes a small diagnostics table decomposing the surface into its
time-invariant and time-varying parts.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401,E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hdf5_min import read_weights  # noqa: E402

DEF_RUN = "runs/estimation/2026-09-08_12-44-40_global_dynamic"
DEF_NODE = "(2, 2, 2)"
OUT = ROOT / "paper/Figures/Projections"


def swish(z):
    return z / (1.0 + np.exp(-z))


def forward(w, X):
    """Chained Dense stack: hidden layers carry a bias, the output layer does not."""
    idx = sorted({int(k.split("/")[1].split("_")[1]) if "_" in k.split("/")[1] else 0
                  for k in w if k.startswith("layers")})
    h = X
    for i in idx:
        b = f"layers/dense{'' if i == 0 else f'_{i}'}/vars"
        h = h @ np.asarray(w[f"{b}/0"], float)
        if f"{b}/1" in w:
            h = swish(h + np.asarray(w[f"{b}/1"], float))
    return h[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEF_RUN)
    ap.add_argument("--node", default=DEF_NODE)
    ap.add_argument("--outfile", default=None)
    a = ap.parse_args()

    d = pd.read_excel(ROOT / "data/MainData.xlsx")
    d = d[d.Year <= 2024]
    Tp = d.pivot(index="Year", columns="CountryCode", values="TempPopWeight")
    Pv = (d.pivot(index="Year", columns="CountryCode", values="PrecipPopWeight")
            .reindex_like(Tp).to_numpy(float))
    Tv = Tp.to_numpy(float)
    years = np.asarray(Tp.index, dtype=int)
    mT, sT = float(np.nanmean(Tv)), float(np.nanstd(Tv))
    mP, sP = float(np.nanmean(Pv)), float(np.nanstd(Pv))
    nT = len(years)
    # time_scaling='standardize': deterministic moments for 1..T
    mt, st = (nT + 1) / 2, np.sqrt((nT ** 2 - 1) / 12)
    Pmed = float(np.nanmedian(Pv))

    w = read_weights(str(ROOT / a.run / "parameters" / f"{a.node}.weights.h5"))

    # Restrict the temperature axis to the supported range: only 3 of 10,324
    # country-years exceed 30 C, so the tails are functional form, not evidence.
    lo, hi = np.nanpercentile(Tv, [5, 95])
    Tg = np.linspace(lo, hi, 160)
    TT, tt = np.meshgrid(Tg, np.arange(1, nT + 1), indexing="ij")
    X = np.stack([(TT - mT) / sT,
                  np.full_like(TT, (Pmed - mP) / sP),
                  (tt - mt) / st], axis=-1)
    F = forward(w, X)
    G = (F - F.mean(axis=0, keepdims=True)) * 100.0        # pp/yr, centred per year

    # decomposition
    Gbar = G.mean(axis=1, keepdims=True)
    R = G - Gbar
    share = float((R ** 2).sum() / (G ** 2).sum())
    amp = G.max(axis=0) - G.min(axis=0)
    corr = float(np.corrcoef(G[:, 0], G[:, -1])[0, 1])

    print(f"run  {a.run}\nnode {a.node}   temperature axis {lo:.1f}-{hi:.1f} C "
          f"(5th-95th pct)   precipitation {Pmed:.0f} mm\n")
    print(f"  time-varying share of surface variation : {share*100:7.2f}%")
    print(f"  time-invariant share                    : {(1-share)*100:7.2f}%")
    print(f"  corr(shape {years[0]}, shape {years[-1]})          : {corr:7.3f}")
    print(f"  amplitude  min {amp.min():.2f}pp ({years[amp.argmin()]})   "
          f"max {amp.max():.2f}pp ({years[amp.argmax()]})   mean {amp.mean():.2f}pp")
    print(f"  time-invariant amplitude                : {Gbar.max()-Gbar.min():7.2f}pp")

    YY = np.tile(years, (len(Tg), 1))
    fig = plt.figure(figsize=(7.2, 5.4))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(TT, YY, G, cmap="cividis", rstride=2, cstride=1,
                    linewidth=0, antialiased=True, shade=True)
    ax.set_xlabel("temperature ($\\degree$C)", labelpad=8)
    ax.set_ylabel("year", labelpad=10)
    ax.set_zlabel("growth response (pp/yr)", labelpad=6)
    ax.view_init(elev=22, azim=-128)
    ax.set_box_aspect((1.25, 1.0, 0.62))
    ax.tick_params(labelsize=8)
    fig.tight_layout()

    OUT.mkdir(parents=True, exist_ok=True)
    dest = Path(a.outfile) if a.outfile else OUT / "dynamic_time_surface.pdf"
    fig.savefig(dest, bbox_inches="tight")
    fig.savefig(dest.with_suffix(".png"), dpi=170, bbox_inches="tight")
    print(f"\nfigure -> {dest}")

    pd.DataFrame(dict(year=years, amplitude_pp=amp,
                      argmax_T=Tg[np.argmax(G, axis=0)])).to_csv(
        ROOT / "paper/Tables/2026-09-08-dynamic-time-variation.csv", index=False)
    print(f"table  -> {ROOT/'paper/Tables/2026-09-08-dynamic-time-variation.csv'}")


if __name__ == "__main__":
    main()
