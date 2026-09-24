"""paper1-analysis figure set for the RESCALED dynamic run.

Run: runs/estimation/2026-09-01_14-04-01_global_dynamic (wb, corrected data,
within projection, dynamic_model=true, time input standardised).

Deviations from the skill's default workflow, both forced:
  * plotly is unavailable in this environment (PyPI blocked), so the figures are
    matplotlib rather than self-contained HTML.
  * the skill's three figures assume a STATIC two-input surface. This run is
    dynamic, so f depends on (T, P, t) and "the surface" is undefined without a
    year. Figures are therefore drawn at the sample midpoint (1992) for the
    architecture comparison, and the BIC-best architecture additionally gets a
    time panel, which is the object of interest for this run.

Surfaces are computed with a pure-NumPy swish forward pass (no TF), reading the
old-style Keras .weights.h5 via scripts/hdf5_min.py. Depth is inferred from the
layer chain, and the third input is the standardised time index
t~ = (t - (T+1)/2) / sqrt((T^2-1)/12) with t = year - 1960, T = 63.

Outputs: results/images/2026-09-01-dynamic/
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
from hdf5_min import read_weights                                    # noqa: E402

RUN = ROOT / "runs/estimation/2026-09-03_12-03-43_global_dynamic"
STATIC = ROOT / ("runs/estimation/2026-08-27_11-01-35_global_IC_country_trends"
                 "/parameters/(2,).weights.h5")
OUT = ROOT / "results/images/2026-09-01-dynamic"
OUT.mkdir(parents=True, exist_ok=True)

MT, ST, MP, SP = 20.3801, 7.2308, 1281.5723, 833.7378
NT = 63
TMU, TSIG = (NT + 1) / 2.0, np.sqrt((NT * NT - 1) / 12.0)
REF_YEAR = 1992
YEARS = [1961, 1980, 2000, 2010, 2020]


def swish(z):
    return z / (1.0 + np.exp(-z))


def layers(path):
    w = read_weights(str(path))
    out, i = [], 0
    while True:
        k = f"layers/dense{'' if i == 0 else '_' + str(i)}/vars/0"
        b = f"layers/dense{'' if i == 0 else '_' + str(i)}/vars/1"
        if k not in w:
            break
        out.append((np.asarray(w[k]), np.asarray(w[b]) if b in w else None))
        i += 1
    return out


def surface(L, T, P, year=None):
    """Forward pass. year=None -> static 2-input net."""
    Tz, Pz = (T - MT) / ST, (P - MP) / SP
    cols = [Tz, Pz]
    if L[0][0].shape[0] == 3:
        t = (year - 1960 - TMU) / TSIG
        cols.append(np.full(np.shape(Tz), t))
    h = np.stack(cols, axis=-1)
    for j, (K, B) in enumerate(L):
        h = h @ K + (0 if B is None else B)
        if j < len(L) - 1:
            h = swish(h)
    return h[..., 0]


def main():
    res = np.load(RUN / "results.npy", allow_pickle=True).item()
    rows = [(str(k), float(v[0]), float(v[1])) for k, v in res.items()
            if isinstance(v, (list, tuple)) and len(v) >= 2]
    df = pd.DataFrame(rows, columns=["node", "BIC", "AIC"]).sort_values("AIC")
    d = df.AIC - df.AIC.min()
    df["akaike_w"] = np.exp(-0.5 * d) / np.exp(-0.5 * d).sum()
    df.to_csv(OUT / "architecture_ranking.csv", index=False)
    eff = 1.0 / (df.akaike_w ** 2).sum()
    print(f"  {len(df)} architectures; AIC-best {df.node.iloc[0]}, "
          f"BIC-best {df.sort_values('BIC').node.iloc[0]}")
    print(f"  top Akaike weight {df.akaike_w.iloc[0]:.3f}; effective model count {eff:.2f}")
    print(df.head(8).to_string(index=False))

    Tv = np.linspace(0, 30, 90)
    Pv = np.linspace(12.03731002, 3000.0, 90)
    Tg, Pg = np.meshgrid(Tv, Pv)
    ci, cj = int(np.argmin(abs(Pv - MP))), int(np.argmin(abs(Tv - MT)))

    # ---- figure 1: top-20 architectures at the reference year -------------
    top = df.head(20)
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    acc = np.zeros_like(Tg)
    for _, r in top.iterrows():
        p = RUN / f"parameters/{r.node}.weights.h5"
        if not p.exists():
            continue
        Z = surface(layers(p), Tg, Pg, REF_YEAR)
        Z = Z - Z[ci, cj]
        acc += r.akaike_w * Z
        ax.plot(Tv, 100 * Z[ci, :], color="#8fa9bf", lw=0.8, alpha=0.65)
    acc /= top.akaike_w.sum()
    ax.plot(Tv, 100 * acc[ci, :], color="#123f5f", lw=2.4,
            label="AIC-weighted mean (top 20)")
    Ls = layers(STATIC)
    Zs = surface(Ls, Tg, Pg); Zs = Zs - Zs[ci, cj]
    ax.plot(Tv, 100 * Zs[ci, :], color="#999999", lw=1.8, ls="--",
            label="static (2,), corrected data")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("annual mean temperature ($\\degree$C)")
    ax.set_ylabel("growth effect, centred at mean climate (pp)")
    ax.set_title(f"A. top-20 architectures by AIC, dynamic run at t={REF_YEAR}",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8.5); ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "top20_surfaces.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "top20_surfaces.pdf", bbox_inches="tight")

    # ---- figure 2: BIC-best architecture, time panel ----------------------
    best = (2, 2, 2)  # BIC-best architecture from the above ranking
    Lb = layers(RUN / f"parameters/{best}.weights.h5")
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(YEARS)))
    stats = []
    for y, c in zip(YEARS, cols):
        Z = surface(Lb, Tg, Pg, y); Z = Z - Z[ci, cj]
        s = 100 * Z[ci, :]
        rng = s.max() - s.min()
        stats.append((y, rng, Tv[int(np.argmax(s))]))
        ax.plot(Tv, s, color=c, lw=2, label=f"{y} (range {rng:.1f} pp)")
    ax.plot(Tv, 100 * Zs[ci, :], color="#999999", lw=1.8, ls="--",
            label="static (2,) reference")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("annual mean temperature ($\\degree$C)")
    ax.set_ylabel("growth effect, centred at mean climate (pp)")
    ax.set_title(f"B. BIC-best dynamic architecture {best}, by year",
                 fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "best_surface_by_year.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "best_surface_by_year.pdf", bbox_inches="tight")

    # ---- figure 3: precipitation percentiles vs Burke ----------------------
    est = pd.read_csv(ROOT / "data/MainData_est.csv", low_memory=False)
    pv = pd.to_numeric(est.PrecipPopWeight, errors="coerce").dropna()
    pcts = np.percentile(pv, [10, 50, 90])
    bT, bT2 = 0.014303, -0.0004914          # Burke refit, corrected data
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharey=True)
    for a, q, lab in zip(axes, pcts, ["10th", "50th", "90th"]):
        j = int(np.argmin(abs(Pv - q)))
        Z = surface(Lb, Tg, Pg, 2020); Z = Z - Z[j, cj]
        a.plot(Tv, 100 * Z[j, :], color="#2f6f9f", lw=2, label=f"dynamic {best}, 2020")
        Zst = surface(Ls, Tg, Pg); Zst = Zst - Zst[j, cj]
        a.plot(Tv, 100 * Zst[j, :], color="#123f5f", lw=1.6, ls="-.", label="static (2,)")
        bq = bT * Tv + bT2 * Tv ** 2
        a.plot(Tv, 100 * (bq - (bT * MT + bT2 * MT ** 2)), color="#b23a48", lw=1.8, ls="--",
               label="Burke quadratic")
        a.axhline(0, color="grey", lw=0.8); a.grid(alpha=0.25, lw=0.6)
        a.set_xlabel("temperature ($\\degree$C)")
        a.set_title(f"{lab} precip pct ({q:.0f} mm)", fontsize=10, loc="left")
        a.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("growth effect (pp)")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("C. temperature cross-sections at precipitation percentiles",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
   
    fig.savefig(OUT / "percentile_best_vs_burke.png", dpi=170, bbox_inches="tight")
    fig.savefig(OUT / "percentile_best_vs_burke.pdf", bbox_inches="tight")

    print(f"\n  BIC-best {best} response range by year:")
    for y, rng, opt in stats:
        edge = " (grid edge - no interior optimum)" if opt <= 0.4 or opt >= 29.6 else ""
        print(f"    {y}: range {rng:5.2f} pp   optimum {opt:5.1f} C{edge}")
    print(f"\n  wrote 3 figures + architecture_ranking.csv to {OUT}")


if __name__ == "__main__":
    main()
