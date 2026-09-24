#!/usr/bin/env python3
"""Reproduce Burke, Hsiang & Miguel (2015) Figure 5a, then swap in our response function.

STEP 1 (validation): with Burke's own coefficients and their own baseline data this
reproduces their headline exactly:

    2099 change in global GDP per capita = -22.77 %      (published: about -23 %)

Three details are essential and easy to miss:
  1. Baselines (meantemp, basegrowth, gdpCap) are averaged over **1980 onward only**,
     not the full 1961-2010 sample (ComputeMainProjections.R).
  2. gdpCap = TotGDP / Pop, computed per row then averaged.
  3. ISO recodes COD->ZAR and ROU->ROM in the SSP files, otherwise Congo DR and
     Romania drop out and you get 163 instead of 165 countries.
Plus: baseline temperature is Burke's UDel population-weighted measure, SSP5 for
population and income, RCP 8.5 country temperature deltas applied linearly, and the
damage function evaluated at min(T, 30 C).

STEP 2 (comparison): the identical harness is then run with
  * Burke's published coefficients          (the validation arm)
  * the Burke quadratic re-estimated on OUR sample
  * the Leirvik interactive spec re-estimated on OUR sample
  * our BIC-selected neural network (2,), evaluated in closed form
so the ONLY thing that varies is the shape of the climate response.

Run from the repo root:  python scripts/2026-08-25-reproduce-burke.py
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
from hdf5_min import read_weights  # noqa: E402

YRS = np.arange(2010, 2100)
TCAP = 30.0
NN_WEIGHTS = ROOT / "runs/estimation/2026-08-27_11-01-35_global_IC_country_trends/parameters/(2,).weights.h5"
OUT = ROOT / "paper/Figures/Projections"
OUT.mkdir(parents=True, exist_ok=True)
C = {"Burke (published)": "#7a8698", "Burke (our sample)": "#b23a48",
     "Leirvik (our sample)": "#c99a2e", "Neural network": "#26426b"}


def ipolate(df, yrs):
    """Annual linear interpolation of the 5-yearly SSP columns (ipolate_like_r)."""
    yc = {int(c): c for c in df.columns if re.fullmatch(r"\d{4}", str(c).strip())}
    sup = np.array(sorted(yc))
    out = np.empty((len(df), len(yrs)))
    for i, y in enumerate(yrs):
        if y in yc:
            out[:, i] = pd.to_numeric(df[yc[y]], errors="coerce")
        else:
            lo, hi = sup[sup < y], sup[sup > y]
            if len(lo) == 0:
                out[:, i] = pd.to_numeric(df[yc[sup.min()]], errors="coerce"); continue
            if len(hi) == 0:
                out[:, i] = pd.to_numeric(df[yc[lo.max()]], errors="coerce"); continue
            yl, yu = lo.max(), hi.min()
            el = pd.to_numeric(df[yc[yl]], errors="coerce").to_numpy()
            eu = pd.to_numeric(df[yc[yu]], errors="coerce").to_numpy()
            out[:, i] = el + (eu - el) * (y - yl) / (yu - yl)
    o = df[["Region"]].reset_index(drop=True).copy()
    o["Region"] = o.Region.replace({"COD": "ZAR", "ROU": "ROM"})     # detail 3
    return pd.concat([o, pd.DataFrame(out, columns=[str(y) for y in yrs])], axis=1)


def within_fit(df, cols):
    """FWL within-OLS: country + year FE and country linear/quadratic trends."""
    cc, yy = pd.Categorical(df.CountryCode), pd.Categorical(df.Year)
    n = len(df)
    D1 = np.zeros((n, len(cc.categories))); D1[np.arange(n), cc.codes] = 1.0
    D2 = np.zeros((n, len(yy.categories))); D2[np.arange(n), yy.codes] = 1.0
    t = df.Year.to_numpy(float); t -= t.mean()
    W = np.hstack([D1, D2, D1 * t[:, None], D1 * (t ** 2)[:, None]])
    B = np.linalg.pinv(W.T @ W) @ W.T
    ann = lambda v: v - W @ (B @ v)
    X = np.column_stack([df[c].to_numpy(float) for c in cols])
    y = df.growth.to_numpy(float)
    Xt = np.column_stack([ann(X[:, k]) for k in range(X.shape[1])])
    return np.linalg.lstsq(Xt, ann(y), rcond=None)[0]


def main():
    # ---------- Burke's own baseline data (details 1 and 2) ----------------
    b = pd.read_csv(ROOT / "data/projections/old/in/mainDataset.csv")
    b["gdpCap"] = b.TotGDP / b.Pop
    b = b[(b.year >= 1980) & b.growthWDI.notna() & b.UDel_temp_popweight.notna()]
    base = (b.groupby("iso", as_index=False)
             .agg(T=("UDel_temp_popweight", "mean"), bg=("growthWDI", "mean"),
                  gdp=("gdpCap", "mean"))
             .rename(columns={"iso": "CountryCode"}))

    # our precipitation baseline (Burke's data has none; precip is held fixed anyway)
    o = pd.read_csv(ROOT / "data/MainData_est.csv")
    o["Pmm"] = pd.to_numeric(o.PrecipPopWeight, errors="coerce")
    prec = (o[o.Year >= 1980].groupby("CountryCode", as_index=False)
              .agg(Pmm=("Pmm", "mean")))
    base = base.merge(prec, on="CountryCode", how="left")
    base["Pmm"] = base.Pmm.fillna(base.Pmm.median())

    # ---------- RCP 8.5 deltas + SSP5 -------------------------------------
    tch = pd.read_csv(ROOT / "data/projections/old/out/CountryTempChange_RCP85.csv",
                      encoding="latin-1")[["GMI_CNTRY", "CNTRY_NAME", "Tchg"]].dropna()
    tch = tch[~tch.CNTRY_NAME.isin({"West Bank", "Gaza Strip", "Bouvet Island"})]
    tch = tch.drop_duplicates("GMI_CNTRY").rename(columns={"GMI_CNTRY": "CountryCode"})

    g = pd.read_csv(ROOT / "data/projections/old/in/SSP/SSP_GrowthProjections.csv", encoding="latin-1")
    p = pd.read_csv(ROOT / "data/projections/old/in/SSP/SSP_PopulationProjections.csv", encoding="latin-1")
    g = g[(g.Model == "OECD Env-Growth") & (g.Scenario == "SSP5_v9_130325")]
    p = p[p.Scenario == "SSP5_v9_130115"]
    gA = ipolate(g, YRS); gA[[str(y) for y in YRS]] /= 100.0
    pA = ipolate(p, YRS)

    m = (base.merge(tch[["CountryCode", "Tchg"]], on="CountryCode")
             .merge(gA.rename(columns={"Region": "CountryCode"}), on="CountryCode")
             .merge(pA.rename(columns={"Region": "CountryCode"}), on="CountryCode",
                    suffixes=("_g", "_p")))
    print(f"countries in projection: {len(m)}   (Burke's replication: 165)")

    t0 = m["T"].to_numpy(float); p0 = m.Pmm.to_numpy(float)
    gdp0 = m.gdp.to_numpy(float); ccd = m.Tchg.to_numpy(float) / len(YRS)
    G_ = m[[f"{y}_g" for y in YRS]].to_numpy(float)
    P_ = m[[f"{y}_p" for y in YRS]].to_numpy(float)

    # ---------- response functions ----------------------------------------
    d = pd.read_csv(ROOT / "data/MainData_est.csv").rename(columns={"GrowthWDI": "growth"})
    d["T"] = pd.to_numeric(d.TempPopWeight, errors="coerce")
    d["Pmm"] = pd.to_numeric(d.PrecipPopWeight, errors="coerce")
    d["Pm"] = d.Pmm / 1000.0
    d = d.dropna(subset=["growth", "T", "Pmm"]).copy()
    d["T2"] = d["T"] ** 2; d["Pm2"] = d.Pm ** 2
    d["TP"] = d["T"] * d.Pm; d["T2P"] = d.T2 * d.Pm
    d["TP2"] = d["T"] * d.Pm2; d["T2P2"] = d.T2 * d.Pm2
    bB = within_fit(d, ["T", "T2", "Pm", "Pm2"])
    bL = within_fit(d, ["T", "T2", "Pm", "Pm2", "TP", "T2P", "TP2", "T2P2"])

    w = read_weights(str(NN_WEIGHTS))
    G = np.asarray(w["layers/dense/vars/0"]); c0 = np.asarray(w["layers/dense/vars/1"])
    bt = np.asarray(w["layers/dense_1/vars/0"]).ravel()
    mT, sT = np.nanmean(d["T"]), np.nanstd(d["T"])
    mP, sP = np.nanmean(d.Pmm), np.nanstd(d.Pmm)

    def f_pub(T, P):     return 0.0127184 * T - 0.0004871 * T ** 2
    def f_burke(T, P):
        Pm = P / 1000.0
        return bB[0]*T + bB[1]*T**2 + bB[2]*Pm + bB[3]*Pm**2
    def f_leir(T, P):
        Pm = P / 1000.0
        return (bL[0]*T + bL[1]*T**2 + bL[2]*Pm + bL[3]*Pm**2
                + bL[4]*T*Pm + bL[5]*T**2*Pm + bL[6]*T*Pm**2 + bL[7]*T**2*Pm**2)
    def f_nn(T, P):
        Tz, Pz = (T - mT) / sT, (P - mP) / sP
        out = 0.0
        for k in (0, 1):
            z = c0[k] + G[0, k] * Tz + G[1, k] * Pz
            out = out + bt[k] * (z / (1.0 + np.exp(-z)))
        return out

    # ---------- Burke's projection loop -----------------------------------
    def project(f):
        cc = np.zeros((len(m), len(YRS))); nc = np.zeros_like(cc)
        cc[:, 0] = gdp0; nc[:, 0] = gdp0
        bg = f(t0, p0); tot = np.zeros((len(YRS), 2))
        for i in range(1, len(YRS)):
            j = i - 1; gy = G_[:, i]
            nc[:, i] = nc[:, j] * (1.0 + gy)
            nt = np.minimum(t0 + j * ccd, TCAP)
            cc[:, i] = cc[:, j] * (1.0 + gy + f(nt, p0) - bg)
            wt = P_[:, i]
            tot[i, 0] = np.average(cc[:, i], weights=wt)
            tot[i, 1] = np.average(nc[:, i], weights=wt)
        chg = np.zeros(len(YRS))
        chg[1:] = (tot[1:, 0] / tot[1:, 1] - 1.0) * 100.0
        return chg

    res = {}
    for name, f in [("Burke (published)", f_pub), ("Burke (our sample)", f_burke),
                    ("Leirvik (our sample)", f_leir), ("Neural network", f_nn)]:
        res[name] = project(f)
        print(f"  {name:22s} 2099: {res[name][-1]:+7.2f}%")
    print(f"\nVALIDATION: Burke published arm = {res['Burke (published)'][-1]:+.2f}% "
          f"(target -22.77%, their paper about -23%)")

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, chg in res.items():
        ax.plot(YRS, chg, color=C[name], lw=2,
                ls="-" if name == "Neural network" else "--", label=name)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("year"); ax.set_ylabel("change in global GDP per capita (%)")
    ax.set_title("RCP 8.5 / SSP5, Burke (2015) harness", fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    for e in ("pdf", "png"):
        fig.savefig(OUT / f"burke_harness.{e}", dpi=160, bbox_inches="tight")
    pd.DataFrame({"Year": YRS, **res}).to_csv(
        ROOT / "results/tables/projection_burke_harness.csv", index=False)
    print(f"wrote {OUT/'burke_harness.pdf'}")


if __name__ == "__main__":
    main()
