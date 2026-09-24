"""RCP 8.5 / SSP5 projections with country-cluster bootstrap confidence bands.

Three arms, all with 95% bands from the SAME resampling design (countries drawn
with replacement, duplicates treated as distinct units):

  Neural network   1000 saved bootstrap surfaces, bilinearly interpolated onto
                   each country's (T, P). No weights were saved by the bootstrap,
                   so the surface grid IS the response function here.
  Burke            1000 country-cluster refits of the quadratic.
  Leirvik          1000 country-cluster refits of the richer polynomial.

Point estimates follow Burke, Hsiang & Miguel (2015): the central line is the
full-sample fit, never a bootstrap mean. The bootstrap supplies only the band.

TEMPERATURE IS CAPPED AT 30 C for every arm, i.e. f(min(T, 30)), exactly as in
BHM's ComputeMainProjections.R ("constrain response to response at 30C ... so we
are not projecting out of sample"). Here it is also forced: the saved bootstrap
surfaces stop at 30 C, and linear extrapolation past the edge was measured to err
by 1.9 pp/yr at 33.8 C, so off-grid extrapolation would badly understate damage.
30.1 C is in any case the warmest value observed in the estimation sample.

Two panels are reported because the population-weighted mean hides the incidence:
under BHM's own published coefficients that aggregate is about -22 % while the
median country is about -77 %.

Outputs: paper/Figures/Projections/projections_bands.{pdf,png}
         paper/Figures/Projections/projections_bands.csv
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
sys.path.insert(0, str(ROOT / "scripts"))
from hdf5_min import read_weights                                   # noqa: E402

BOOT = ROOT / "runs/bootstrap/2026-08-29_20-56-38global_IC"
PEST = ROOT / ("runs/estimation/2026-08-27_11-01-35_global_IC_country_trends"
               "/parameters/(2,).weights.h5")
OUT = ROOT / "paper/Figures/Projections"
OUT.mkdir(parents=True, exist_ok=True)

YRS = list(range(2010, 2100))
TCAP = 30.0
N_BOOT_Q = 1000
MT, ST, MP, SP = 20.3801, 7.2308, 1281.5723, 833.7378
C = {"Neural network": "#2f6f9f", "Burke (our sample)": "#b23a48",
     "Leirvik (our sample)": "#e08a3c", "Burke (published)": "#7a8698"}

# reuse the validated harness (within_fit, ipolate, load_ours, quad)
_src = (ROOT / "scripts/2026-08-28-projections-corrected.py").read_text(encoding="utf-8")
_ns = {"__name__": "h",
       "__file__": str(ROOT / "scripts/2026-08-28-projections-corrected.py")}
exec(compile(_src.replace('if __name__ == "__main__":', 'if False:'), "h", "exec"), _ns)
within_fit, ipolate, load_ours, quad = (_ns["within_fit"], _ns["ipolate"],
                                        _ns["load_ours"], _ns["quad"])


def bilinear_weights(vals, grid):
    """Index/weight pairs for linear interpolation of vals onto a 1-D grid."""
    v = np.clip(vals, grid[0], grid[-1])
    i1 = np.clip(np.searchsorted(grid, v), 1, len(grid) - 1)
    i0 = i1 - 1
    w1 = (v - grid[i0]) / (grid[i1] - grid[i0])
    return i0, i1, 1.0 - w1, w1


def main():
    t_start = time.time()

    # ---------------- inputs (identical to the validated harness) -------------
    b = pd.read_csv(ROOT / "data/projections/old/in/mainDataset.csv")
    b["gdpCap"] = b.TotGDP / b.Pop
    bb = b[(b.year >= 1980) & b.growthWDI.notna() & b.UDel_temp_popweight.notna()]
    base = (bb.groupby("iso", as_index=False)
              .agg(gdp=("gdpCap", "mean")).rename(columns={"iso": "CountryCode"}))

    d = load_ours()
    est = d.dropna(subset=["growth", "T", "Pmm"]).copy()
    ag = (est[est.Year >= 1980].groupby("iso", as_index=False)
            .agg(Tours=("T", "mean"), Pmm=("Pmm", "mean"))
            .rename(columns={"iso": "CountryCode"}))
    base = base.merge(ag, on="CountryCode", how="inner")

    tch = pd.read_csv(ROOT / "data/projections/old/out/CountryTempChange_RCP85.csv",
                      encoding="latin-1")[["GMI_CNTRY", "CNTRY_NAME", "Tchg"]].dropna()
    tch = tch[~tch.CNTRY_NAME.isin({"West Bank", "Gaza Strip", "Bouvet Island"})]
    tch = tch.drop_duplicates("GMI_CNTRY").rename(columns={"GMI_CNTRY": "CountryCode"})

    g = pd.read_csv(ROOT / "data/projections/old/in/SSP/SSP_GrowthProjections.csv",
                    encoding="latin-1")
    p = pd.read_csv(ROOT / "data/projections/old/in/SSP/SSP_PopulationProjections.csv",
                    encoding="latin-1")
    g = g[(g.Model == "OECD Env-Growth") & (g.Scenario == "SSP5_v9_130325")]
    p = p[p.Scenario == "SSP5_v9_130115"]
    gA = ipolate(g, YRS); gA[[str(y) for y in YRS]] /= 100.0
    pA = ipolate(p, YRS)

    m = (base.merge(tch[["CountryCode", "Tchg"]], on="CountryCode")
             .merge(gA.rename(columns={"Region": "CountryCode"}), on="CountryCode")
             .merge(pA.rename(columns={"Region": "CountryCode"}), on="CountryCode",
                    suffixes=("_g", "_p")))
    n_c = len(m)
    t0 = m.Tours.to_numpy(float); p0 = m.Pmm.to_numpy(float)
    gdp0 = m.gdp.to_numpy(float); ccd = m.Tchg.to_numpy(float) / len(YRS)
    G_ = m[[f"{y}_g" for y in YRS]].to_numpy(float)
    P_ = m[[f"{y}_p" for y in YRS]].to_numpy(float)
    print(f"countries: {n_c}   draws: NN=1000, quadratics={N_BOOT_Q}")

    # temperature path, capped at 30 C for every arm
    Tpath = np.minimum(t0[:, None] + np.arange(len(YRS))[None, :] * ccd[:, None], TCAP)
    Tbase = np.minimum(t0, TCAP)

    # ---------------- NN: interpolate the 1000 saved surfaces -----------------
    S = np.load(BOOT / "bootstrap_surfaces.npy")[:, 0].astype(np.float64)   # (B,90,90)
    zz = np.load(BOOT / "bootstrap_bands.npz", allow_pickle=True)
    Tg, Pg = zz["T"][0, :], zz["P"][:, 0]
    B = S.shape[0]
    pj0, pj1, pw0, pw1 = bilinear_weights(p0, Pg)                # per country
    rows = S[:, pj0, :] * pw0[None, :, None] + S[:, pj1, :] * pw1[None, :, None]
    #      rows: (B, n_c, 90) = f as a function of the T grid, at each country's P

    def nn_f(Tvals):
        """Tvals (n_c, k) -> (B, n_c, k) interpolated along T."""
        i0, i1, w0, w1 = bilinear_weights(Tvals.ravel(), Tg)
        i0 = i0.reshape(Tvals.shape); i1 = i1.reshape(Tvals.shape)
        w0 = w0.reshape(Tvals.shape); w1 = w1.reshape(Tvals.shape)
        ci = np.arange(Tvals.shape[0])[:, None]
        return rows[:, ci, i0] * w0[None] + rows[:, ci, i1] * w1[None]

    nn_base = nn_f(Tbase[:, None])[:, :, 0]                      # (B, n_c)
    nn_path = nn_f(Tpath)                                        # (B, n_c, 90)
    dg_nn = nn_path - nn_base[:, :, None]
    del S, rows, nn_path

    # point estimate from the full-sample weights
    w = read_weights(str(PEST))
    K = np.asarray(w["layers/dense/vars/0"]); c0 = np.asarray(w["layers/dense/vars/1"])
    bt = np.asarray(w["layers/dense_1/vars/0"]).ravel()

    def f_nn_pt(T, P):
        Tz, Pz = (T - MT) / ST, (P - MP) / SP
        o = 0.0
        for k in range(K.shape[1]):
            z = c0[k] + K[0, k] * Tz + K[1, k] * Pz
            o = o + bt[k] * (z / (1.0 + np.exp(-z)))
        return o

    dg_nn_pt = (f_nn_pt(Tpath, p0[:, None]) - f_nn_pt(Tbase, p0)[:, None])

    # ---------------- quadratics: 1000 country-cluster refits -----------------
    e = est.copy()
    e["Pm"] = e.Pmm / 1000.0; e["T2"] = e["T"] ** 2; e["Pm2"] = e.Pm ** 2
    e["TP"] = e["T"] * e.Pm; e["T2P"] = e.T2 * e.Pm
    e["TP2"] = e["T"] * e.Pm2; e["T2P2"] = e.T2 * e.Pm2
    QB = ["T", "T2", "Pm", "Pm2"]
    QL = ["T", "T2", "Pm", "Pm2", "TP", "T2P", "TP2", "T2P2"]
    bB_pt, bL_pt = within_fit(e, QB), within_fit(e, QL)

    z = np.load(ROOT / "data/bootstrap_quad_coefs.npz")
    coefB, coefL = z["burke"], z["leirvik"]
    print(f"  loaded cached quadratic bootstrap: {coefB.shape[0]} draws "
          f"(scripts/2026-08-30-quad-bootstrap.py)")

    def quad_dg(coef, poly):
        """coef (R,k) -> (R, n_c, 90) growth deltas.

        Precipitation is held at baseline, so every precip-only term cancels in
        f(newT, P) - f(baseT, P); only the temperature and the temperature x
        precipitation interactions survive.
        """
        Pm = (p0 / 1000.0)[None, :, None]
        dT = (Tpath - Tbase[:, None])[None]
        dT2 = (Tpath ** 2 - (Tbase ** 2)[:, None])[None]
        a = coef[:, 0][:, None, None]
        c2 = coef[:, 1][:, None, None]
        out = a * dT + c2 * dT2
        if poly == "L":                      # order: T,T2,Pm,Pm2,TP,T2P,TP2,T2P2
            out = (out
                   + coef[:, 4][:, None, None] * dT * Pm
                   + coef[:, 5][:, None, None] * dT2 * Pm
                   + coef[:, 6][:, None, None] * dT * Pm ** 2
                   + coef[:, 7][:, None, None] * dT2 * Pm ** 2)
        return out

    dg_B = quad_dg(coefB[~np.isnan(coefB).any(1)], "B")
    dg_L = quad_dg(coefL[~np.isnan(coefL).any(1)], "L")
    dg_B_pt = quad_dg(bB_pt[None, :], "B")[0]
    dg_L_pt = quad_dg(bL_pt[None, :], "L")[0]
    print(f"  Burke {dg_B.shape[0]} draws ok, Leirvik {dg_L.shape[0]} draws ok")

    # ---------------- Burke's compounding projection --------------------------
    def project(dg):
        """dg (R, n_c, 90) -> aggregate %, median country %, pop-share-harmed %."""
        R = dg.shape[0]
        cc = np.tile(gdp0, (R, 1)); nc = gdp0.copy()
        agg = np.zeros((R, len(YRS))); med = np.zeros((R, len(YRS)))
        hrm = np.zeros((R, len(YRS)))
        for i in range(1, len(YRS)):
            gy = G_[:, i]
            nc = nc * (1.0 + gy)
            # dg[..., k] is evaluated at Tpath[:, k] = t0 + k*ccd, so year i takes
            # dg[..., i]: BHM's 1-based loop uses j = i-1 as BOTH the previous
            # array slot and the warming step, and only the former shifts when
            # translated to 0-based indexing (newtemp = temp + j*ccd, j = 1..89).
            cc = cc * (1.0 + gy[None, :] + dg[:, :, i])
            wt = P_[:, i]
            imp = cc / nc[None, :] - 1.0
            agg[:, i] = (np.average(cc, weights=wt, axis=1)
                         / np.average(nc, weights=wt) - 1.0) * 100.0
            med[:, i] = np.median(imp, axis=1) * 100.0
            hrm[:, i] = (imp < 0) @ wt / wt.sum() * 100.0
        return agg, med, hrm

    arms = {}
    for nm, dgb, dgp in [("Neural network", dg_nn, dg_nn_pt[None]),
                         ("Burke (our sample)", dg_B, dg_B_pt[None]),
                         ("Leirvik (our sample)", dg_L, dg_L_pt[None])]:
        a, md, h = project(dgb)
        ap, mp, hp = project(dgp)
        arms[nm] = dict(agg=a, med=md, hrm=h, agg_pt=ap[0], med_pt=mp[0], hrm_pt=hp[0])
        lo, hi = np.percentile(a[:, -1], [2.5, 97.5])
        print(f"  {nm:22s} 2099 aggregate {ap[0,-1]:+7.2f}%  [{lo:+.1f}, {hi:+.1f}]"
              f"   median country {mp[0,-1]:+7.1f}%")

    # ---------------- figure ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    for nm, r in arms.items():
        for ax, key, pt in ((axes[0], "agg", "agg_pt"), (axes[1], "med", "med_pt")):
            lo = np.percentile(r[key], 2.5, axis=0); hi = np.percentile(r[key], 97.5, axis=0)
            ax.fill_between(YRS, lo, hi, color=C[nm], alpha=0.16, lw=0)
            ax.plot(YRS, r[pt], color=C[nm], lw=2.0, label=nm)
    for ax, ttl, yl in ((axes[0], "A. population-weighted mean GDP per capita",
                         "change in global GDP per capita (%)"),
                        (axes[1], "B. median country", "change in GDP per capita (%)")):
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("year"); ax.set_ylabel(yl)
        ax.set_title(ttl, fontsize=10.5, loc="left")
        ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25, lw=0.6)
    # Panel A is clipped: the NN upper band runs to about +317% by 2099, which
    # would flatten every point estimate. Burke clips his own Fig 5a to -75/+60
    # for the same reason. The overflow is annotated rather than hidden.
    top = 100.0
    over = {nm: np.percentile(r["agg"], 97.5, axis=0)[-1] for nm, r in arms.items()}
    axes[0].set_ylim(-75, top)
    axes[0].annotate(
        "upper 97.5% at 2099 (off-scale):\n"
        + "\n".join(f"  {nm.split(' (')[0]}  {v:+.0f}%" for nm, v in over.items()),
        xy=(0.03, 0.97), xycoords="axes fraction", va="top", fontsize=7.5,
        color="#444444")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower left")
    fig.suptitle("RCP 8.5 / SSP5 with 95% country-cluster bootstrap bands "
                 "(response capped at 30 $\\degree$C)", fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"projections_bands.{ext}", dpi=170, bbox_inches="tight")

    rec = {"year": YRS}
    for nm, r in arms.items():
        k = nm.split()[0].lower()
        rec[f"{k}_agg"] = r["agg_pt"]
        rec[f"{k}_agg_lo"] = np.percentile(r["agg"], 2.5, axis=0)
        rec[f"{k}_agg_hi"] = np.percentile(r["agg"], 97.5, axis=0)
        rec[f"{k}_median_country"] = r["med_pt"]
        rec[f"{k}_pop_harmed"] = r["hrm_pt"]
    pd.DataFrame(rec).to_csv(OUT / "projections_bands.csv", index=False)
    print(f"\ntotal {(time.time()-t_start)/60:.1f} min -> {OUT/'projections_bands.pdf'}")


if __name__ == "__main__":
    main()
