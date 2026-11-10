"""Open the hood on the selected global architecture: hidden-layer coefficients,
the learned factors A1 and A2, and how much each loads on the climate index.

The BIC-selected global model is (2,): one hidden layer, two swish neurons, and
a linear output with no bias. Written out, with standardised inputs,

    NN_i,nt = c_i + gamma_i1 * Tstd_nt + gamma_i2 * Pstd_nt
    A_i,nt  = g(NN_i,nt),            g = swish
    f_nt    = beta_1 A_1,nt + beta_2 A_2,nt

and f enters the growth equation on top of country fixed effects, time fixed
effects and country trends, which the within projection absorbs.

Four panels:
  (a) A1 and A2 against temperature, at mean precipitation
  (b) A1 and A2 against precipitation, at mean temperature
  (c) the contributions beta_i A_i and their sum, against temperature
  (d) cross-country means by year, standardised, against mean T and P

Reading note for (a) and (b): A1 and A2 live on scales that differ by a factor
of about 45, so they are drawn on twin axes. That difference is the point of
panel (c), where both are converted to the common units of the climate index and
can share one axis.

Usage
-----
    python scripts/2026-10-02-hidden-layer-factors.py
    python scripts/2026-10-02-hidden-layer-factors.py --node "(2,)" --run-dir ...
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from scripts.archive.hdf5_min import read_weights

RUN = REPO / "runs/estimation/2026-08-27_11-01-35_global_IC_country_trends"
OUT = REPO / "paper/Figures/Hidden"

C_PRECIP = "#2171b5"
C_TEMP = "#d94801"
C_TOTAL = "black"
FS = 11

plt.rcParams.update({
    "font.size": FS,
    "axes.labelsize": FS,
    "axes.titlesize": FS + 1,
    "xtick.labelsize": FS - 1,
    "ytick.labelsize": FS - 1,
    "legend.fontsize": FS - 1,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def swish(z):
    return z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def dswish(z):
    s = 1.0 / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))
    return s * (1.0 + z * (1.0 - s))


def swish_minimum():
    """Argmin of swish, found on a dense grid to avoid a scipy dependency."""
    z = np.linspace(-5.0, -0.5, 500001)
    return float(z[np.argmin(np.abs(dswish(z)))])


def load_model(run_dir, node):
    w = read_weights(str(Path(run_dir) / "parameters" / f"{node}.weights.h5"))
    W1 = np.asarray(w["layers/dense/vars/0"], float)
    b1 = np.asarray(w["layers/dense/vars/1"], float)
    beta = np.asarray(w["layers/dense_1/vars/0"], float).ravel()
    if W1.shape != (2, 2) or beta.shape != (2,):
        raise SystemExit(f"this script assumes the (2,) architecture; got W1 {W1.shape}, beta {beta.shape}")
    return W1, b1, beta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default=str(RUN))
    ap.add_argument("--node", default="(2,)")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    W1, b1, beta = load_model(args.run_dir, args.node)

    data = pd.read_excel(REPO / "data/MainData.xlsx")
    MT, ST = float(np.nanmean(data["TempPopWeight"])), float(np.nanstd(data["TempPopWeight"]))
    MP, SP = float(np.nanmean(data["PrecipPopWeight"])), float(np.nanstd(data["PrecipPopWeight"]))

    ok = (data["TempPopWeight"].notna() & data["PrecipPopWeight"].notna()
          & data["GrowthWDI"].notna())
    panel = data.loc[ok, ["Year", "CountryCode", "TempPopWeight", "PrecipPopWeight"]].copy()

    def factors(T, P):
        X = np.column_stack([(np.asarray(T, float) - MT) / ST,
                             (np.asarray(P, float) - MP) / SP])
        return swish(X @ W1 + b1)

    A = factors(panel["TempPopWeight"], panel["PrecipPopWeight"])
    contrib = A * beta
    index = A @ beta

    # ---------------------------------------------------------------- table
    zmin = swish_minimum()
    t_star = ((zmin - b1[1]) / W1[0, 1]) * ST + MT

    rows = [{
        "neuron": i + 1,
        "constant": b1[i],
        "gamma_T_std": W1[0, i],
        "gamma_P_std": W1[1, i],
        "dNN_dT_per_C": W1[0, i] / ST,
        "dNN_dP_per_m": W1[1, i] / SP * 1000.0,
        "temp_share": abs(W1[0, i]) / (abs(W1[0, i]) + abs(W1[1, i])),
        "beta": beta[i],
        "sd_A": A[:, i].std(),
        "sd_contribution": contrib[:, i].std(),
        "var_share": contrib[:, i].var() / index.var(),
    } for i in range(2)]
    table = pd.DataFrame(rows)
    table.to_csv(out / "hidden_layer_coefficients.csv", index=False)

    v1, v2 = W1[:, 0], W1[:, 1]
    cos = float(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2)))

    print(f"run   {Path(args.run_dir).name}")
    print(f"node  {args.node}      n = {len(panel)} country-years\n")
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.5f}"))
    print(f"\nangle between the two input directions: "
          f"{np.degrees(np.arccos(np.clip(cos, -1, 1))):.2f} deg (cos {cos:+.4f})")
    print(f"swish minimum at z = {zmin:.5f}")
    print(f"temperature neuron crosses it at T = {t_star:.2f} C")
    print(f"sd(f) = {index.std():.5f}   "
          f"corr(f, f without A1) = {np.corrcoef(index, contrib[:, 1])[0, 1]:.4f}")

    # --------------------------------------------------------------- figure
    # One panel per factor, each against the input it specialises in. The two
    # neurons are orthogonal to within 0.7 degrees, so showing A2 against
    # precipitation (or A1 against temperature) would only draw a near-flat
    # line; the specialisation itself is reported in the table above.
    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(11.0, 4.3))

    t_grid = np.linspace(0.0, 32.0, 400)
    p_grid = np.linspace(0.0, 3500.0, 400)

    # (a) the temperature factor against temperature, at mean precipitation
    A_t = factors(t_grid, np.full_like(t_grid, MP))
    ax_t.plot(t_grid, A_t[:, 1], color=C_TEMP, lw=2.2)
   
    ax_t.set_xlabel("Temperature (\u00b0C)")
    ax_t.set_ylabel("$A_2$")
    ax_t.set_title(rf"(a) temperature factor ", loc="left")

    # (b) the precipitation factor against precipitation, at mean temperature
    A_p = factors(np.full_like(p_grid, MT), p_grid)
    p_star = p_grid[int(np.argmin(A_p[:, 0]))]
    ax_p.plot(p_grid / 1000.0, A_p[:, 0], color=C_PRECIP, lw=2.2)
   
    ax_p.set_xlabel("Precipitation (m)")
    ax_p.set_ylabel("$A_1$")
    ax_p.set_title(rf"(b) precipitation factor", loc="left")

    print(f"precipitation factor turns at P = {p_star:.0f} mm")

    fig.tight_layout()
    pdf = out / "hidden_layer_factors.pdf"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(out / "hidden_layer_factors.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------- figure 2: weight space
    # The two input-weight vectors, normalised to unit length, drawn in the
    # (temperature, precipitation) weight plane.
    #
    # Both axes carry weights on STANDARDISED inputs, so the plane is isotropic
    # and the angle between the vectors is a meaningful quantity rather than an
    # artefact of the units of degrees and millimetres. Equal aspect is
    # therefore required, not cosmetic: without it a right angle would not be
    # drawn as one. Normalising to unit length isolates direction from
    # magnitude, which differs by a factor of 3.7 between the two columns.
    fig2, ax = plt.subplots(figsize=(5.4, 5.4))

    v1, v2 = W1[:, 0], W1[:, 1]
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    cos12 = float(v1 @ v2 / (n1 * n2))
    ang = float(np.degrees(np.arccos(np.clip(cos12, -1.0, 1.0))))
    u1, u2 = v1 / n1, v2 / n2

    circ = np.linspace(0.0, 2.0 * np.pi, 400)
    ax.plot(np.cos(circ), np.sin(circ), color="0.85", lw=0.9, zorder=0)
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
    ax.axvline(0.0, color="0.75", lw=0.8, zorder=0)

    for u, col in ((u1, C_PRECIP), (u2, C_TEMP)):
        ax.annotate("", xy=(u[0], u[1]), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4,
                                    shrinkA=0, shrinkB=0))

    a1 = np.degrees(np.arctan2(u1[1], u1[0]))
    a2 = np.degrees(np.arctan2(u2[1], u2[0]))
    arc = np.radians(np.linspace(min(a1, a2), max(a1, a2), 200))
    ax.plot(0.38 * np.cos(arc), 0.38 * np.sin(arc), color="0.45", lw=1.1)
    mid = np.radians(0.5 * (a1 + a2))
    ax.text(0.50 * np.cos(mid), 0.50 * np.sin(mid), f"{ang:.1f}$^\\circ$",
            color="0.35", ha="center", va="center", fontsize=FS)

    ax.text(u1[0], u1[1] + 0.10, r"$\gamma_1$ (precipitation)", color=C_PRECIP,
            ha="center", fontsize=FS)
    ax.text(u2[0] + 0.04, u2[1] - 0.12, r"$\gamma_2$ (temperature)", color=C_TEMP,
            ha="right", va="top", fontsize=FS)

    ax.set_xlim(-1.35, 1.35)
    ax.set_ylim(-1.35, 1.35)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("weight on temperature")
    ax.set_ylabel("weight on precipitation")
    
    fig2.tight_layout()
    pdf2 = out / "hidden_layer_vectors.pdf"
    fig2.savefig(pdf2, bbox_inches="tight")
    fig2.savefig(out / "hidden_layer_vectors.png", dpi=160, bbox_inches="tight")
    plt.close(fig2)

    print(f"\n-> {pdf}")
    print(f"-> {out / 'hidden_layer_factors.png'}")
    print(f"-> {pdf2}")
    print(f"-> {out / 'hidden_layer_vectors.png'}")
    print(f"-> {out / 'hidden_layer_coefficients.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
