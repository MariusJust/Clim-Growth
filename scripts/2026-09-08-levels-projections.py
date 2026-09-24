"""Levels-model response surface and RCP 8.5 / SSP5 projections.

The levels specification models log GDP per capita (Kalkuhl-Wenz setup), so the
climate net returns a LEVEL effect, not a growth rate. Impacts therefore do not
compound:

    impact_i,t = exp( f(T_i,t, P_i) - f(T_i,base, P_i) ) - 1
    GDPcc_i,t  = GDPnocc_i,t * exp( f(T_i,t, P_i) - f(T_i,base, P_i) )

Everything else follows Burke, Hsiang & Miguel (2015) ComputeMainProjections.R:
ccd = Tchg/90, newtemp = temp + j*ccd for j = 1..89, response pinned at 30 C
(D10), baseline temperature = country mean 1980+.

Reporting follows D16: median country and population share harmed lead, Burke's
ratio of population-weighted means is shown alongside, and the leave-out-winners
row is the robustness check.

Architecture: (2,), BIC-best in the levels run and decisively so (next best is
43.5 BIC away).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hdf5_min import read_weights  # noqa: E402

YRS = list(range(2010, 2100))
TCAP = 30.0
LEV = ROOT / ("runs/estimation/2026-09-04_11-55-57_global_IC_country_trends_levels"
              "/parameters/(2,).weights.h5")
GRO = ROOT / ("runs/estimation/2026-08-27_11-01-35_global_IC_country_trends"
              "/parameters/(2,).weights.h5")
OUT = ROOT / "paper/Tables"
OUT.mkdir(parents=True, exist_ok=True)

_src = (ROOT / "scripts/2026-08-28-projections-corrected.py").read_text(encoding="utf-8")
_ns = {"__name__": "h", "__file__": str(ROOT / "scripts/2026-08-28-projections-corrected.py")}
exec(compile(_src.replace('if __name__ == "__main__":', 'if False:'), "h", "exec"), _ns)
ipolate, load_ours = _ns["ipolate"], _ns["load_ours"]


def net(path):
    w = read_weights(str(path))
    K = np.asarray(w["layers/dense/vars/0"], float)
    c0 = np.asarray(w["layers/dense/vars/1"], float)
    bt = np.asarray(w["layers/dense_1/vars/0"], float).ravel()

    def f(T, P, mT, sT, mP, sP):
        Tz, Pz = (np.asarray(T, float) - mT) / sT, (np.asarray(P, float) - mP) / sP
        o = 0.0
        for j in range(K.shape[1]):
            z = c0[j] + K[0, j] * Tz + K[1, j] * Pz
            o = o + bt[j] * (z / (1.0 + np.exp(-z)))
        return o
    return f


def main():
    md = pd.read_excel(ROOT / "data/MainData.xlsx")
    md = md[md.Year <= 2024]
    Tv = md.pivot(index="Year", columns="CountryCode", values="TempPopWeight").to_numpy(float)
    Pv = md.pivot(index="Year", columns="CountryCode", values="PrecipPopWeight").to_numpy(float)
    mT, sT = float(np.nanmean(Tv)), float(np.nanstd(Tv))
    mP, sP = float(np.nanmean(Pv)), float(np.nanstd(Pv))
    f_lev, f_gro = net(LEV), net(GRO)

    # ---------------- response surface ---------------------------------------
    Pmed = float(np.nanmedian(Pv))
    Tg = np.linspace(0, 30, 3001)
    yl = f_lev(Tg, Pmed, mT, sT, mP, sP); yl = yl - yl[int(np.argmin(np.abs(Tg - mT)))]
    yg = f_gro(Tg, Pmed, mT, sT, mP, sP); yg = yg - yg[int(np.argmin(np.abs(Tg - mT)))]
    print(f"Response at median precipitation ({Pmed:.0f} mm), centred at the "
          f"sample mean {mT:.2f} C\n")
    print(f"  {'T (C)':>7}{'levels: % effect on GDP pc':>30}{'growth: pp/yr':>18}")
    for t in (5, 10, 12, 15, 20, 25, 30):
        i = int(np.argmin(np.abs(Tg - t)))
        print(f"  {t:7.0f}{(np.exp(yl[i])-1)*100:29.2f}%{yg[i]*100:17.2f}")
    print(f"\n  optimum   levels {Tg[int(np.argmax(yl))]:.2f} C   "
          f"growth {Tg[int(np.argmax(yg))]:.2f} C   "
          f"Burke published 13.0 C, Burke on our data 14.55 C")

    # ---------------- projection inputs --------------------------------------
    b = pd.read_csv(ROOT / "data/projections/old/in/mainDataset.csv")
    b["gdpCap"] = b.TotGDP / b.Pop
    bb = b[(b.year >= 1980) & b.growthWDI.notna() & b.UDel_temp_popweight.notna()]
    base = (bb.groupby("iso", as_index=False).agg(gdp=("gdpCap", "mean"))
              .rename(columns={"iso": "CountryCode"}))
    dd = load_ours()
    est = dd.dropna(subset=["growth", "T", "Pmm"]).copy()
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
    t0 = m.Tours.to_numpy(float); p0 = m.Pmm.to_numpy(float)
    gdp0 = m.gdp.to_numpy(float); ccd = m.Tchg.to_numpy(float) / len(YRS)
    G_ = m[[f"{y}_g" for y in YRS]].to_numpy(float)
    P_ = m[[f"{y}_p" for y in YRS]].to_numpy(float)
    wt = P_[:, -1]
    iso = m.CountryCode.to_numpy()

    # counterfactual (no climate change) path, common to both arms
    nc = gdp0.copy()
    for i in range(1, len(YRS)):
        nc = nc * (1.0 + G_[:, i])

    Tend = np.minimum(t0 + (len(YRS) - 1) * ccd, TCAP)
    Tbase = np.minimum(t0, TCAP)
    dlog = (f_lev(Tend, p0, mT, sT, mP, sP) - f_lev(Tbase, p0, mT, sT, mP, sP))
    imp = np.exp(dlog) - 1.0                       # level effect, no compounding
    cc = nc * np.exp(dlog)

    agg = (np.average(cc, weights=wt) / np.average(nc, weights=wt) - 1) * 100
    contrib = wt * (cc - nc) / np.sum(wt * nc) * 100
    o = np.argsort(-contrib)

    def agg_excl(k):
        keep = np.ones(len(m), bool); keep[o[:k]] = False
        return (np.sum(wt[keep]*cc[keep])/np.sum(wt[keep]*nc[keep]) - 1)*100

    print(f"\n\n=== Levels projection, 2099, SSP5 / RCP 8.5, {len(m)} countries ===")
    print(f"  (level effect, no compounding)\n")
    print(f"  HEADLINE (D16)")
    print(f"    median country                       : {np.median(imp)*100:+9.2f}%")
    print(f"    world population worse off           : {wt[imp<0].sum()/wt.sum()*100:9.1f}%")
    print(f"    countries worse off                  : {(imp<0).mean()*100:9.1f}%")
    print(f"\n  alongside, for comparability")
    print(f"    Burke ratio of pop-weighted means    : {agg:+9.2f}%")
    print(f"    pop-weighted mean of country impacts : "
          f"{np.average(imp, weights=wt)*100:+9.2f}%")
    print(f"    25th / 75th percentile country       : "
          f"{np.percentile(imp,25)*100:+8.2f}% / {np.percentile(imp,75)*100:+.2f}%")
    print(f"\n  robustness: aggregate after removing the largest winners")
    print(f"    all {agg:+7.2f}%   drop top 1 {agg_excl(1):+7.2f}%   "
          f"drop top 3 {agg_excl(3):+7.2f}%   drop top 5 {agg_excl(5):+7.2f}%")
    print(f"\n  largest contributors")
    print(f"    {'iso':>5}{'impact':>10}{'contrib':>10}")
    for k in list(o[:3]) + list(o[-3:]):
        print(f"    {iso[k]:>5}{imp[k]*100:+9.1f}%{contrib[k]:+9.2f}%")

    pd.DataFrame(dict(CountryCode=iso, impact_pct=imp*100, contrib_pp=contrib,
                      pop2099_m=wt, Tbase=Tbase, Tend=Tend)).to_csv(
        OUT / "2026-09-08-levels-projections.csv", index=False)
    print(f"\ntable -> {OUT/'2026-09-08-levels-projections.csv'}")


if __name__ == "__main__":
    main()
