"""RCP 8.5 / SSP5 out-of-sample projections on the CORRECTED climate data.

Runs Burke's (2015) projection harness with four response functions:
    * Burke's published coefficients        (validation arm, must give -22.77 %)
    * Burke's quadratic re-estimated on our corrected data
    * Leirvik's richer polynomial on our corrected data
    * the neural network, run 2026-08-27_11-01-35 (BIC-best (2,))

It then reports an ATTRIBUTION LADDER that decomposes the move from Burke's
published -22.77 % to the re-estimated number into
    (i) WDI growth-data vintage, (ii) corrected climate data, (iii) sample period.

Finally it checks sensitivity to using our own baseline temperatures instead of
Burke's UDel baseline for the arms estimated on our data.

Outputs: paper/Figures/Projections/projections_corrected.{pdf,png}
         paper/Figures/Projections/projections_corrected.csv
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

YRS = list(range(2010, 2100))
TCAP = 30.0
RUN = "2026-08-27_11-01-35_global_IC_country_trends"
NN_WEIGHTS = ROOT / f"runs/estimation/{RUN}/parameters/(2,).weights.h5"
DATA_END = 2024                                                     # from run config
OUT = ROOT / "paper/Figures/Projections"
OUT.mkdir(parents=True, exist_ok=True)

C = {"Burke (published)": "#7a8698", "Burke (our sample)": "#b23a48",
     "Leirvik (our sample)": "#e08a3c", "Neural network": "#2f6f9f"}


# ----------------------------------------------------------------- helpers --
def within_fit(df, cols):
    """FWL within-OLS: country + year FE, country linear and quadratic trends."""
    cc, yy = pd.Categorical(df.CountryCode), pd.Categorical(df.Year)
    n = len(df)
    D1 = np.zeros((n, len(cc.categories))); D1[np.arange(n), cc.codes] = 1.0
    D2 = np.zeros((n, len(yy.categories))); D2[np.arange(n), yy.codes] = 1.0
    t = df.Year.to_numpy(float); t -= t.mean()
    W = np.hstack([D1, D2, D1 * t[:, None], D1 * (t ** 2)[:, None]])
    B = np.linalg.pinv(W.T @ W) @ W.T
    ann = lambda v: v - W @ (B @ v)
    X = np.column_stack([df[c].to_numpy(float) for c in cols])
    Xt = np.column_stack([ann(X[:, k]) for k in range(X.shape[1])])
    return np.linalg.lstsq(Xt, ann(df.growth.to_numpy(float)), rcond=None)[0]


def ipolate(df, yrs):
    yc = {}
    for c in df.columns:
        s = str(c).strip()
        if s[:4].isdigit() and len(s) >= 4:
            yc[int(s[:4])] = c
    ys = np.array(sorted(yc))
    out = np.empty((len(df), len(yrs)))
    for i, y in enumerate(yrs):
        if y in yc:
            out[:, i] = pd.to_numeric(df[yc[y]], errors="coerce"); continue
        lo, hi = ys[ys < y], ys[ys > y]
        if len(lo) == 0:
            out[:, i] = pd.to_numeric(df[yc[hi.min()]], errors="coerce"); continue
        if len(hi) == 0:
            out[:, i] = pd.to_numeric(df[yc[lo.max()]], errors="coerce"); continue
        yl, yu = lo.max(), hi.min()
        el = pd.to_numeric(df[yc[yl]], errors="coerce").to_numpy()
        eu = pd.to_numeric(df[yc[yu]], errors="coerce").to_numpy()
        out[:, i] = el + (eu - el) * (y - yl) / (yu - yl)
    o = df[["Region"]].reset_index(drop=True).copy()
    o["Region"] = o.Region.replace({"COD": "ZAR", "ROU": "ROM"})
    return pd.concat([o, pd.DataFrame(out, columns=[str(y) for y in yrs])], axis=1)


def load_ours():
    d = pd.read_csv(ROOT / "data/MainData_est.csv", low_memory=False)
    d = d.rename(columns={"GrowthWDI": "growth"})
    d["T"] = pd.to_numeric(d.TempPopWeight, errors="coerce")
    d["Pmm"] = pd.to_numeric(d.PrecipPopWeight, errors="coerce")
    d["growth"] = pd.to_numeric(d.growth, errors="coerce")
    d["iso"] = d.CountryCode.replace({"COD": "ZAR", "ROU": "ROM"})
    return d


def quad(bb):
    """Burke-style quadratic (precip terms cancel: precipitation is held fixed)."""
    def f(T, P):
        Pm = P / 1000.0
        z = bb[0] * T + bb[1] * T ** 2
        if len(bb) >= 4:
            z = z + bb[2] * Pm + bb[3] * Pm ** 2
        return z
    return f


def main():
    # ---------- Burke's baseline (his UDel temps, 1980+, gdpCap = TotGDP/Pop) --
    b = pd.read_csv(ROOT / "data/projections/old/in/mainDataset.csv")
    b["gdpCap"] = b.TotGDP / b.Pop
    bb = b[(b.year >= 1980) & b.growthWDI.notna() & b.UDel_temp_popweight.notna()]
    base = (bb.groupby("iso", as_index=False)
              .agg(T=("UDel_temp_popweight", "mean"), gdp=("gdpCap", "mean"))
              .rename(columns={"iso": "CountryCode"}))

    d = load_ours()
    est = d.dropna(subset=["growth", "T", "Pmm"]).copy()
    prec = (est[est.Year >= 1980].groupby("iso", as_index=False)
              .agg(Pmm=("Pmm", "mean")).rename(columns={"iso": "CountryCode"}))
    ourT = (est[est.Year >= 1980].groupby("iso", as_index=False)
              .agg(Tours=("T", "mean")).rename(columns={"iso": "CountryCode"}))
    base = base.merge(prec, on="CountryCode", how="left").merge(ourT, on="CountryCode", how="left")
    base["Pmm"] = base.Pmm.fillna(base.Pmm.median())
    base["Tours"] = base.Tours.fillna(base["T"])

    # ---------- RCP 8.5 warming + SSP5 growth/population ----------------------
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
    print(f"countries in projection: {len(m)}   (Burke's replication: 165)\n")

    t0 = m["T"].to_numpy(float)
    t0_ours = m.Tours.to_numpy(float)
    p0 = m.Pmm.to_numpy(float)
    gdp0 = m.gdp.to_numpy(float)
    ccd = m.Tchg.to_numpy(float) / len(YRS)
    G_ = m[[f"{y}_g" for y in YRS]].to_numpy(float)
    P_ = m[[f"{y}_p" for y in YRS]].to_numpy(float)

    def project(f, tbase=None, cap=TCAP):
        tb = t0 if tbase is None else tbase
        cc = np.zeros((len(m), len(YRS))); nc = np.zeros_like(cc)
        cc[:, 0] = gdp0; nc[:, 0] = gdp0
        bg = f(tb, p0); tot = np.zeros((len(YRS), 2))
        for i in range(1, len(YRS)):
            # BHM's ComputeMainProjections.R runs a 1-based loop where j = i-1 is
            # BOTH the previous array slot AND the number of years of warming.
            # Those coincide only in 1-based indexing: here the previous slot is
            # i-1 but the warming step is i, matching his newtemp = temp + j*ccd
            # for j = 1..89.
            j = i - 1; gy = G_[:, i]
            nc[:, i] = nc[:, j] * (1.0 + gy)
            nt = tb + i * ccd if cap is None else np.minimum(tb + i * ccd, cap)
            cc[:, i] = cc[:, j] * (1.0 + gy + f(nt, p0) - bg)
            wt = P_[:, i]
            tot[i, 0] = np.average(cc[:, i], weights=wt)
            tot[i, 1] = np.average(nc[:, i], weights=wt)
        chg = np.zeros(len(YRS))
        chg[1:] = (tot[1:, 0] / tot[1:, 1] - 1.0) * 100.0
        return chg

    # ---------- response functions -------------------------------------------
    e = est.copy()
    e["Pm"] = e.Pmm / 1000.0; e["T2"] = e["T"] ** 2; e["Pm2"] = e.Pm ** 2
    e["TP"] = e["T"] * e.Pm; e["T2P"] = e.T2 * e.Pm
    e["TP2"] = e["T"] * e.Pm2; e["T2P2"] = e.T2 * e.Pm2
    bB = within_fit(e, ["T", "T2", "Pm", "Pm2"])
    bL = within_fit(e, ["T", "T2", "Pm", "Pm2", "TP", "T2P", "TP2", "T2P2"])

    w = read_weights(str(NN_WEIGHTS))
    K = np.asarray(w["layers/dense/vars/0"]); c0 = np.asarray(w["layers/dense/vars/1"])
    bt = np.asarray(w["layers/dense_1/vars/0"]).ravel()
    # standardisation exactly as in training: Year x Country pivot, Year <= data_end
    tr = d[d.Year <= DATA_END]
    pT = tr.pivot_table(index="Year", columns="CountryCode", values="T", aggfunc="mean")
    pP = tr.pivot_table(index="Year", columns="CountryCode", values="Pmm", aggfunc="mean")
    mT, sT = np.nanmean(pT.values), np.nanstd(pT.values)
    mP, sP = np.nanmean(pP.values), np.nanstd(pP.values)

    def f_pub(T, P):
        return 0.0127184 * T - 0.0004871 * T ** 2

    def f_leir(T, P):
        Pm = P / 1000.0
        return (bL[0]*T + bL[1]*T**2 + bL[2]*Pm + bL[3]*Pm**2
                + bL[4]*T*Pm + bL[5]*T**2*Pm + bL[6]*T*Pm**2 + bL[7]*T**2*Pm**2)

    def f_nn(T, P):
        Tz, Pz = (T - mT) / sT, (P - mP) / sP
        out = 0.0
        for k in range(K.shape[1]):
            z = c0[k] + K[0, k] * Tz + K[1, k] * Pz
            out = out + bt[k] * (z / (1.0 + np.exp(-z)))
        return out

    # Each response function is evaluated at the baseline temperature of the
    # dataset it was estimated on: Burke's published coefficients come from UDel,
    # so they keep the UDel baseline; our arms use our corrected baseline.
    arms = [("Burke (published)", f_pub, None), ("Burke (our sample)", quad(bB), t0_ours),
            ("Leirvik (our sample)", f_leir, t0_ours), ("Neural network", f_nn, t0_ours)]
    # headline: NO temperature cap (warming is not truncated at 30 C)
    res = {n: project(f, tb, cap=None) for n, f, tb in arms}
    res_cap = {n: project(f, tb, cap=TCAP) for n, f, tb in arms}

    tmax_u = (t0_ours + (len(YRS) - 1) * ccd)
    print("=== 2099 change in global GDP per capita, RCP 8.5 / SSP5 ===")
    print("    (each arm evaluated at its own baseline temperatures)")
    print(f"{'arm':24s}{'UNCAPPED':>12}{'capped 30C':>13}{'diff':>9}")
    for n, _, _ in arms:
        print(f"  {n:22s}{res[n][-1]:+11.2f}%{res_cap[n][-1]:+12.2f}%"
              f"{res[n][-1]-res_cap[n][-1]:+8.1f}")
    print(f"\n2099 country temperatures without the cap: "
          f"max {tmax_u.max():.1f} C, mean {tmax_u.mean():.1f} C, "
          f"{(tmax_u > 30).sum()} of {len(tmax_u)} countries above 30 C")
    hot = est["T"].max()
    print(f"warmest temperature ever OBSERVED in the estimation sample: {hot:.1f} C"
          f"  ->  {(tmax_u > hot).sum()} countries extrapolate beyond it by 2099")

    # ---------- distribution behind the aggregate -----------------------------
    # Burke's headline is a population-weighted mean of GDP/cap, which is
    # dominated by a few cold countries whose gains compound. Reporting the
    # country distribution alongside it is essential: under Burke's OWN published
    # coefficients the median country is about -76 % while the aggregate is -22 %.
    def paths(f, tbase=None, cap=None):
        tb = t0 if tbase is None else tbase
        cc = np.zeros((len(m), len(YRS))); nc = np.zeros_like(cc)
        cc[:, 0] = gdp0; nc[:, 0] = gdp0
        bg = f(tb, p0)
        for i in range(1, len(YRS)):
            # previous array slot is i-1; years of warming is i (see project())
            j = i - 1; gy = G_[:, i]
            nc[:, i] = nc[:, j] * (1.0 + gy)
            nt = tb + i * ccd if cap is None else np.minimum(tb + i * ccd, cap)
            cc[:, i] = cc[:, j] * (1.0 + gy + f(nt, p0) - bg)
        return (cc / nc - 1.0) * 100.0

    print("\n=== what the aggregate hides (2099) ===")
    print(f"{'arm':24s}{'aggregate':>11}{'median ctry':>13}{'ctry hurt':>11}{'pop hurt':>10}")
    for n, f, tb in arms:
        im = paths(f, tb)[:, -1]
        wpop = P_[:, -1]
        print(f"  {n:22s}{res[n][-1]:+10.2f}%{np.median(im):+12.1f}%"
              f"{(im < 0).sum():>7}/{len(im):<4}"
              f"{100*wpop[im < 0].sum()/wpop.sum():9.0f}%")
    print(f"\nVALIDATION  Burke published = {res['Burke (published)'][-1]:+.2f} % "
          f"(target -22.77 %)")

    # ---------- attribution ladder -------------------------------------------
    bo = b.rename(columns={"iso": "CountryCode", "year": "Year",
                           "growthWDI": "growth", "UDel_temp_popweight": "T"})
    bo = bo.dropna(subset=["growth", "T"]).copy(); bo["T2"] = bo["T"] ** 2
    steps = []
    steps.append(("Burke data, Burke sample (published spec)",
                  within_fit(bo, ["T", "T2"])))
    mg = est[est.Year <= 2010].merge(
        bo[["CountryCode", "Year", "T", "growth"]].rename(columns={"T": "Tu", "growth": "gu"}),
        left_on=["iso", "Year"], right_on=["CountryCode", "Year"], how="inner",
        suffixes=("", "_b"))
    a = mg.copy(); a["T"] = a.Tu; a["T2"] = a["T"] ** 2; a["growth"] = a.gu
    steps.append(("  + our row set (UDel temps, Burke growth)", within_fit(a, ["T", "T2"])))
    a2 = mg.copy(); a2["T"] = a2.Tu; a2["T2"] = a2["T"] ** 2
    steps.append(("  + our WDI growth vintage", within_fit(a2, ["T", "T2"])))
    a3 = mg.copy(); a3["T2"] = a3["T"] ** 2
    steps.append(("  + our CORRECTED climate data", within_fit(a3, ["T", "T2"])))
    a4 = est.copy(); a4["T2"] = a4["T"] ** 2
    steps.append(("  + full period to 2024 / all countries", within_fit(a4, ["T", "T2"])))

    print("\n=== attribution: what moves Burke's -22.8 % ? (T + T^2 spec) ===")
    print(f"{'specification':46s}{'b_T':>10}{'b_T2':>12}{'T*':>7}{'2099':>9}")
    prev = None
    for tag, bq in steps:
        v = project(quad(bq), cap=None)[-1]
        star = -bq[0] / (2 * bq[1])
        dl = "" if prev is None else f"   ({v - prev:+.1f} pp)"
        print(f"{tag:46s}{bq[0]:+10.5f}{bq[1]:+12.6f}{star:7.2f}{v:+9.2f}%{dl}")
        prev = v

    # ---------- estimation-window sensitivity (COVID) -------------------------
    # Burke estimates on 1960-2010. Our sample runs to 2024 and so contains the
    # 2020-21 COVID growth collapse, which year FE only partly absorb.
    print("\n=== estimation window (our Burke arm, own baseline) ===")
    for tag, sub in [("full sample to 2024", e),
                     ("excl. 2020-2021", e[~e.Year.isin([2020, 2021])]),
                     ("to 2019 only", e[e.Year <= 2019]),
                     ("to 2010 (Burke window)", e[e.Year <= 2010])]:
        bq = within_fit(sub, ["T", "T2", "Pm", "Pm2"])
        v = project(quad(bq), t0_ours, cap=None)[-1]
        print(f"  {tag:24s} T*={-bq[0]/(2*bq[1]):5.2f}  2099 {v:+7.2f}%  N={len(sub):,}")

    # ---------- baseline-consistency check -----------------------------------
    print("\n=== sensitivity: own baseline temps for our-sample arms ===")
    print(f"  mean baseline  UDel {t0.mean():.2f} C   ours {t0_ours.mean():.2f} C")
    for n, f, _ in arms[1:]:
        print(f"  {n:22s} UDel base {project(f, cap=None)[-1]:+8.2f} %   "
              f"own base {project(f, t0_ours, cap=None)[-1]:+8.2f} %")

    # ---------- outputs -------------------------------------------------------
    pd.DataFrame({"year": YRS, **{n: res[n] for n, _, _ in arms},
                  **{n + " [capped30]": res_cap[n] for n, _, _ in arms}}).to_csv(
        OUT / "projections_corrected.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for n, _, _ in arms:
        ax.plot(YRS, res[n], color=C[n], lw=2,
                ls="-" if n == "Neural network" else "--", label=n)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_xlabel("year"); ax.set_ylabel("change in global GDP per capita (%)")
    ax.set_title("RCP 8.5 / SSP5, corrected climate data (no temperature cap)",
                 fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False); ax.grid(alpha=0.25, lw=0.6)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"projections_corrected.{ext}", dpi=160, bbox_inches="tight")
    print(f"\nwrote {OUT/'projections_corrected.pdf'}")


if __name__ == "__main__":
    main()
