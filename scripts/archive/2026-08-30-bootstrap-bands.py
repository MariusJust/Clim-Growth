"""Response-function figure with country-cluster bootstrap confidence bands.

Reads the 1000-draw bootstrap (runs/bootstrap/2026-08-29_20-56-38global_IC) and
plots the temperature cross-section of the climate response surface at mean
precipitation, centred so that f(mean climate) = 0, with the 2.5/97.5 pointwise
band and the full-sample point estimate on top.

The point estimate is NOT a bootstrap quantity: following Burke, Hsiang & Miguel
(2015), the central line is the full-sample fit (here the saved (2,) weights of
the 2026-08-27 corrected-data run) and the bootstrap supplies only the band.

Outputs: paper/Figures/Bootstrap/response_band.{pdf,png}
         paper/Figures/Bootstrap/response_band.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hdf5_min import read_weights                                   # noqa: E402

BOOT = ROOT / "runs/bootstrap/2026-08-29_20-56-38global_IC"
PEST = ROOT / ("runs/estimation/2026-08-27_11-01-35_global_IC_country_trends"
               "/parameters/(2,).weights.h5")
OUT = ROOT / "paper/Figures/Bootstrap"
OUT.mkdir(parents=True, exist_ok=True)

# full-sample standardisation (Year x Country pivot, Year <= 2024)
MT, ST, MP, SP = 20.3801, 7.2308, 1281.5723, 833.7378


def point_surface(T, P):
    w = read_weights(str(PEST))
    K = np.asarray(w["layers/dense/vars/0"])
    c0 = np.asarray(w["layers/dense/vars/1"])
    bt = np.asarray(w["layers/dense_1/vars/0"]).ravel()
    Tz, Pz = (T - MT) / ST, (P - MP) / SP
    out = 0.0
    for k in range(K.shape[1]):
        z = c0[k] + K[0, k] * Tz + K[1, k] * Pz
        out = out + bt[k] * (z / (1.0 + np.exp(-z)))
    return out


def main():
    S = np.load(BOOT / "bootstrap_surfaces.npy")[:, 0]        # (B, 90, 90)
    z = np.load(BOOT / "bootstrap_bands.npz", allow_pickle=True)
    T, P = z["T"], z["P"]
    pi, ti = z["center_idx"]
    n_ok = int((~np.isnan(S).all(axis=(1, 2))).sum())
    print(f"{n_ok}/{S.shape[0]} draws | centre cell T={T[pi,ti]:.2f} C P={P[pi,ti]:.0f} mm")

    C = S - S[:, pi, ti][:, None, None]                       # centre each draw
    PE = point_surface(T, P)
    PE = PE - PE[pi, ti]

    Tv = T[pi, :]
    pe = PE[pi, :]
    md = np.nanmedian(C[:, pi, :], axis=0)
    lo = np.nanpercentile(C[:, pi, :], 2.5, axis=0)
    hi = np.nanpercentile(C[:, pi, :], 97.5, axis=0)

    # ---- Burke quadratics, centred on the SAME cell so the curves are comparable
    # Precipitation is constant along this cross-section, so Burke's precip terms
    # cancel in the centring and only b_T, b_T2 matter.
    T0 = T[pi, ti]
    _h = (ROOT / "scripts/2026-08-28-projections-corrected.py").read_text(encoding="utf-8")
    _n = {"__name__": "h",
          "__file__": str(ROOT / "scripts/2026-08-28-projections-corrected.py")}
    exec(compile(_h.replace('if __name__ == "__main__":', 'if False:'), "h", "exec"), _n)
    d = _n["load_ours"]()
    e = d.dropna(subset=["growth", "T", "Pmm"]).copy()
    e["Pm"] = e.Pmm / 1000.0
    e["T2"] = e["T"] ** 2; e["Pm2"] = e.Pm ** 2
    bq = _n["within_fit"](e, ["T", "T2", "Pm", "Pm2"])
    print(f"Burke refit on corrected data: b_T={bq[0]:+.6f} b_T2={bq[1]:+.7f} "
          f"T*={-bq[0]/(2*bq[1]):.2f} C")

    def cen_quad(bT, bT2, Tvals):
        return (bT * Tvals + bT2 * Tvals ** 2) - (bT * T0 + bT2 * T0 ** 2)

    burke_ours = cen_quad(bq[0], bq[1], T[pi, :])
    burke_pub = cen_quad(0.0127184, -0.0004871, T[pi, :])

    opt = Tv[np.nanargmax(C[:, pi, :], axis=1)]
    o_lo, o_md, o_hi = np.percentile(opt, [2.5, 50, 97.5])
    interior = float(np.mean((opt > Tv[1]) & (opt < Tv[-2])) * 100)
    print(f"optimum: point {Tv[np.argmax(pe)]:.2f} C | boot median {o_md:.2f} "
          f"[{o_lo:.2f}, {o_hi:.2f}] | interior optimum in {interior:.1f}% of draws")

    # Does the quadratic lie inside the network's 95% band? This is the
    # inferential question the overlay is there to answer.
    for tag, cur in (("Burke, our data", burke_ours), ("Burke, published", burke_pub)):
        inside = (cur >= lo) & (cur <= hi)
        out_T = Tv[~inside]
        frac = 100.0 * inside.mean()
        span = (f"outside at {out_T.min():.1f}-{out_T.max():.1f} C"
                if out_T.size else "inside everywhere")
        print(f"  {tag:18s} inside the NN 95% band over {frac:5.1f}% of 0-30 C  ({span})")

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.fill_between(Tv, 100 * lo, 100 * hi, color="#2f6f9f", alpha=0.20,
                    lw=0, label="95% cluster-bootstrap band")
    ax.plot(Tv, 100 * md, color="#2f6f9f", lw=1.0, ls=":", label="bootstrap median")
    ax.plot(Tv, 100 * pe, color="#123f5f", lw=2.2, label="NN point estimate (full sample)")
    ax.plot(Tv, 100 * burke_ours, color="#b23a48", lw=1.8, ls="--",
            label=f"Burke quadratic, our data (T*={-bq[0]/(2*bq[1]):.1f} $\\degree$C)")
    ax.plot(Tv, 100 * burke_pub, color="#7a8698", lw=1.5, ls=":",
            label="Burke published (T*=13.1 $\\degree$C)")
    ax.axhline(0, color="grey", lw=0.8)
    ax.axvline(Tv[np.argmax(pe)], color="#123f5f", lw=1.0, ls="-.", alpha=0.55,
               label=f"NN optimum {Tv[np.argmax(pe)]:.1f} $\\degree$C")
    ax.set_xlabel("annual mean temperature ($\\degree$C)")
    ax.set_ylabel("growth effect, centred at mean climate (pp)")
    ax.set_title("Climate response function, 1000 country-cluster bootstrap draws",
                 fontsize=11, loc="left")
    ax.set_xlim(0, 30)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"response_band.{e}", dpi=170, bbox_inches="tight")

    np.savetxt(OUT / "response_band.csv",
               np.column_stack([Tv, pe, md, lo, hi, burke_ours, burke_pub]),
               delimiter=",",
               header=("temperature_C,nn_point_estimate,nn_boot_median,nn_lo2.5,"
                       "nn_hi97.5,burke_our_data,burke_published"),
               comments="")
    print(f"wrote {OUT/'response_band.pdf'}")


if __name__ == "__main__":
    main()
