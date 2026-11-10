#!/usr/bin/env python3
"""RCP 8.5 / SSP5 out-of-sample projections, Burke (2015) setup.

Runs three response functions through the *identical* projection machinery:

  * NN        - the BIC-selected global network (2,), evaluated in closed form
                from the saved weights (no TensorFlow needed)
  * Burke     - quadratic  T + T^2 + P + P^2
  * Leirvik   - Burke plus the four interaction terms T*P, T^2*P, T*P^2, T^2*P^2

Both benchmarks are re-estimated on OUR sample with the same within
transformation (country + year fixed effects, country linear and quadratic
trends), so the only thing that differs across the three arms is the shape of
the climate response.

Projection algorithm follows Burke's replication exactly (see
Benchmark/projections_Burke.py, which is a port of their R code):

    ccd      = Tchg / n_years                      (annual increment)
    newtemp  = temp + j * ccd,  capped at 30 C     (their extrapolation guard)
    diff     = f(newtemp, P_base) - f(temp, P_base)
    GDPcc    = GDPcc  * (1 + g_ssp + diff)
    GDPnocc  = GDPnocc* (1 + g_ssp)
    chg      = pop-weighted mean GDPcc / pop-weighted mean GDPnocc - 1

Precipitation is held at each country's baseline (Burke's experiment varies
temperature only); the precipitation variant is a separate run.

Run from the repo root:  python scripts/2026-08-25-project-rcp85.py
"""
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, "/sessions/charming-admiring-fermat/mnt/.claude/skills/paper1-analysis/scripts")
from hdf5_min import read_weights  # noqa: E402

YRS = np.arange(2010, 2100)
TCAP = 30.0                      # Burke's damage-function temperature cap
OUT = ROOT / "paper/Figures/Projections"
OUT.mkdir(parents=True, exist_ok=True)
DARKBLUE, BORDEAUX, GOLD = "#26426b", "#b23a48", "#c99a2e"


# ----------------------------------------------------------------- within OLS
def within_fit(df, cols):
    """FWL: regress growth on `cols` after annihilating country/year FE and
    country linear+quadratic trends. Returns the climate coefficients."""
    cc = pd.Categorical(df.CountryCode)
    yy = pd.Categorical(df.Year)
    n = len(df)
    D1 = np.zeros((n, len(cc.categories))); D1[np.arange(n), cc.codes] = 1.0
    D2 = np.zeros((n, len(yy.categories))); D2[np.arange(n), yy.codes] = 1.0
    t = df.Year.to_numpy(float); t = t - t.mean()
    W = np.hstack([D1, D2, D1 * t[:, None], D1 * (t ** 2)[:, None]])
    B = np.linalg.pinv(W.T @ W) @ W.T

    def annih(v):
        return v - W @ (B @ v)

    X = np.column_stack([df[c].to_numpy(float) for c in cols])
    y = df.growth.to_numpy(float)
    Xt = np.column_stack([annih(X[:, k]) for k in range(X.shape[1])])
    yt = annih(y)
    beta = np.linalg.lstsq(Xt, yt, rcond=None)[0]
    r2 = 1 - np.sum((y - (W @ (B @ y) + X @ beta - W @ (B @ (X @ beta)))) ** 2) / \
        np.sum((y - y.mean()) ** 2)
    return beta, r2


# ------------------------------------------------------------ SSP annual grid
def ipolate(df, yrs):
    ycols = {int(c): c for c in df.columns if re.fullmatch(r"\d{4}", str(c).strip())}
    sup = np.array(sorted(ycols))
    out = np.empty((len(df), len(yrs)))
    for i, y in enumerate(yrs):
        if y in ycols:
            out[:, i] = pd.to_numeric(df[ycols[y]], errors="coerce")
        else:
            lo = sup[sup < y]; hi = sup[sup > y]
            if len(lo) == 0:
                out[:, i] = pd.to_numeric(df[ycols[sup.min()]], errors="coerce"); continue
            if len(hi) == 0:
                out[:, i] = pd.to_numeric(df[ycols[lo.max()]], errors="coerce"); continue
            yl, yu = lo.max(), hi.min()
            el = pd.to_numeric(df[ycols[yl]], errors="coerce").to_numpy()
            eu = pd.to_numeric(df[ycols[yu]], errors="coerce").to_numpy()
            out[:, i] = el + (eu - el) * (y - yl) / (yu - yl)
    res = df[["Region"]].reset_index(drop=True).copy()
    return pd.concat([res, pd.DataFrame(out, columns=[str(y) for y in yrs])], axis=1)


def main():
    # ---------------- our estimation sample ------------------------------
    d = pd.read_csv(ROOT / "data/MainData_est.csv")
    d = d.rename(columns={"GrowthWDI": "growth"})
    d["T"] = pd.to_numeric(d.TempPopWeight, errors="coerce")
    d["Pmm"] = pd.to_numeric(d.PrecipPopWeight, errors="coerce")
    d["Pm"] = d.Pmm / 1000.0
    d = d.dropna(subset=["growth", "T", "Pmm"]).copy()
    d["T2"] = d["T"] ** 2; d["Pm2"] = d.Pm ** 2
    for a, b in [("TP", ("T", "Pm")), ("T2P", ("T2", "Pm")),
                 ("TP2", ("T", "Pm2")), ("T2P2", ("T2", "Pm2"))]:
        d[a] = d[b[0]] * d[b[1]]
    print(f"sample: {len(d):,} obs, {d.CountryCode.nunique()} countries, "
          f"{int(d.Year.min())}-{int(d.Year.max())}")

    bB, r2B = within_fit(d, ["T", "T2", "Pm", "Pm2"])
    bL, r2L = within_fit(d, ["T", "T2", "Pm", "Pm2", "TP", "T2P", "TP2", "T2P2"])
    print(f"Burke   within R2={r2B:.4f}  coef={np.round(bB,6)}")
    print(f"Leirvik within R2={r2L:.4f}  coef={np.round(bL,6)}")

    # ---------------- NN closed form -------------------------------------
    w = read_weights(str(ROOT / "runs/estimation/2026-06-13_22-11-06_global_IC_country_trends/parameters/(2,).weights.h5"))
    G = np.asarray(w["layers/dense/vars/0"]); c0 = np.asarray(w["layers/dense/vars/1"])
    bt = np.asarray(w["layers/dense_1/vars/0"]).ravel()
    mT, sT = np.nanmean(d["T"]), np.nanstd(d["T"])
    mP, sP = np.nanmean(d.Pmm), np.nanstd(d.Pmm)

    def f_nn(T, Pmm):
        Tz = (T - mT) / sT; Pz = (Pmm - mP) / sP
        out = 0.0
        for k in (0, 1):
            z = c0[k] + G[0, k] * Tz + G[1, k] * Pz
            out = out + bt[k] * (z / (1.0 + np.exp(-z)))
        return out

    def f_burke(T, Pmm):
        Pm = Pmm / 1000.0
        return bB[0]*T + bB[1]*T**2 + bB[2]*Pm + bB[3]*Pm**2

    def f_leirvik(T, Pmm):
        Pm = Pmm / 1000.0
        return (bL[0]*T + bL[1]*T**2 + bL[2]*Pm + bL[3]*Pm**2
                + bL[4]*T*Pm + bL[5]*T**2*Pm + bL[6]*T*Pm**2 + bL[7]*T**2*Pm**2)

    # ---------------- country baselines ----------------------------------
    base = d.groupby("CountryCode").agg(T=("T", "mean"), Pmm=("Pmm", "mean"),
                                        gdpCap=("GDPCap", "mean")).reset_index()

    # ---------------- Burke's RCP 8.5 deltas + SSP5 ----------------------
    tch = pd.read_csv(ROOT / "data/projections/old/out/CountryTempChange_RCP85.csv", encoding="latin-1")
    tch = tch[["GMI_CNTRY", "CNTRY_NAME", "Tchg"]].dropna()
    tch = tch[~tch.CNTRY_NAME.isin({"West Bank", "Gaza Strip", "Bouvet Island"})]
    tch = tch.drop_duplicates("GMI_CNTRY").rename(columns={"GMI_CNTRY": "CountryCode"})

    g = pd.read_csv(ROOT / "data/projections/old/in/SSP/SSP_GrowthProjections.csv", encoding="latin-1")
    p = pd.read_csv(ROOT / "data/projections/old/in/SSP/SSP_PopulationProjections.csv", encoding="latin-1")
    g = g[(g.Model == "OECD Env-Growth") & (g.Scenario == "SSP5_v9_130325")]
    p = p[p.Scenario == "SSP5_v9_130115"]
    gA = ipolate(g, YRS); gA[[str(y) for y in YRS]] /= 100.0     # % -> rate
    pA = ipolate(p, YRS)

    m = (base.merge(tch[["CountryCode", "Tchg"]], on="CountryCode")
             .merge(gA.rename(columns={"Region": "CountryCode"}), on="CountryCode")
             .merge(pA.rename(columns={"Region": "CountryCode"}), on="CountryCode",
                    suffixes=("_g", "_p")))
    print(f"countries in projection: {len(m)}")

    temp0 = m["T"].to_numpy(float); prec0 = m.Pmm.to_numpy(float)
    gdp0 = m.gdpCap.to_numpy(float); ccd = m.Tchg.to_numpy(float) / len(YRS)
    G_ = m[[f"{y}_g" for y in YRS]].to_numpy(float)
    P_ = m[[f"{y}_p" for y in YRS]].to_numpy(float)

    # ---------------- projection loop ------------------------------------
    def project(f):
        cc = np.zeros((len(m), len(YRS))); nc = np.zeros_like(cc)
        cc[:, 0] = gdp0; nc[:, 0] = gdp0
        base_eff = f(temp0, prec0)
        tot = np.zeros((len(YRS), 2))
        for i in range(1, len(YRS)):
            j = i - 1
            gy = G_[:, i]
            nc[:, i] = nc[:, j] * (1.0 + gy)
            newt = np.minimum(temp0 + j * ccd, TCAP)
            diff = f(newt, prec0) - base_eff
            cc[:, i] = cc[:, j] * (1.0 + gy + diff)
            wt = P_[:, i]
            tot[i, 0] = np.average(cc[:, i], weights=wt)
            tot[i, 1] = np.average(nc[:, i], weights=wt)
        chg = np.zeros(len(YRS))
        chg[1:] = (tot[1:, 0] / tot[1:, 1] - 1.0) * 100.0
        return chg, cc, nc

    res = {}
    for name, f in [("Neural network", f_nn), ("Burke quadratic", f_burke),
                    ("Leirvik interactive", f_leirvik)]:
        chg, cc, nc = project(f)
        res[name] = chg
        print(f"{name:22s} 2099 change in global GDP per capita: {chg[-1]:+8.2f}%")

    print(f"\ndifference NN - Burke at 2099: "
          f"{res['Neural network'][-1] - res['Burke quadratic'][-1]:+.2f} pp")

    # ---------------- figure ---------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5))
    for (name, chg), col, ls in zip(res.items(), [DARKBLUE, BORDEAUX, GOLD],
                                    ["-", "--", ":"]):
        ax.plot(YRS, chg, color=col, ls=ls, lw=2, label=name)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("year")
    ax.set_ylabel("change in global GDP per capita (%)")
    ax.set_title("RCP 8.5 / SSP5 projection, relative to a no-climate-change baseline",
                 fontsize=11, loc="left")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"projection_rcp85.{ext}", dpi=160, bbox_inches="tight")
    pd.DataFrame({"Year": YRS, **res}).to_csv(
        ROOT / "results/tables/projection_rcp85.csv", index=False)
    print(f"\nwrote {OUT/'projection_rcp85.pdf'}")


if __name__ == "__main__":
    main()
