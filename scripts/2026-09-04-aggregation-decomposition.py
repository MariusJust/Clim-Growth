"""Why the population-weighted aggregate is positive while the median country is
deeply negative.

Burke's headline statistic is a RATIO OF POPULATION-WEIGHTED MEANS,

    agg = sum_i w_i * GDPcc_i / sum_i w_i * GDPnocc_i - 1,

which rearranges to an exactly additive country decomposition,

    agg = sum_i [ w_i * (GDPcc_i - GDPnocc_i) / sum_j w_j * GDPnocc_j ],

so each country's contribution to the headline number can be read off directly.
That is the object this script computes.  The point is that the ratio of means
is not the mean of the ratios: it is a level-weighted statistic, so a country
that compounds a small positive growth-rate difference for 89 years enters the
numerator with an enormous absolute dollar gain regardless of how few people
live there, while a poor country losing 70% of its income barely moves it.

Sections:
  A  distribution of country impacts vs the headline aggregate
  B  additive decomposition: which countries carry the aggregate
  C  aggregate after removing the largest winners
  D  the same projection summarised by five different aggregators

Read-only apart from the CSV it writes to paper/Tables/.
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
MT, ST, MP, SP = 20.3801, 7.2308, 1281.5723, 833.7378
OUT = ROOT / "paper/Tables"
OUT.mkdir(parents=True, exist_ok=True)

_src = (ROOT / "scripts/2026-08-28-projections-corrected.py").read_text(encoding="utf-8")
_ns = {"__name__": "h", "__file__": str(ROOT / "scripts/2026-08-28-projections-corrected.py")}
exec(compile(_src.replace('if __name__ == "__main__":', 'if False:'), "h", "exec"), _ns)
within_fit, ipolate, load_ours = _ns["within_fit"], _ns["ipolate"], _ns["load_ours"]


def build_inputs():
    b = pd.read_csv(ROOT / "data/projections/old/in/mainDataset.csv")
    b["gdpCap"] = b.TotGDP / b.Pop
    bb = b[(b.year >= 1980) & b.growthWDI.notna() & b.UDel_temp_popweight.notna()]
    base = (bb.groupby("iso", as_index=False).agg(gdp=("gdpCap", "mean"))
              .rename(columns={"iso": "CountryCode"}))

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

    m = (base.merge(tch[["CountryCode", "CNTRY_NAME"]], on="CountryCode")
             .merge(tch[["CountryCode", "Tchg"]], on="CountryCode")
             .merge(gA.rename(columns={"Region": "CountryCode"}), on="CountryCode")
             .merge(pA.rename(columns={"Region": "CountryCode"}), on="CountryCode",
                    suffixes=("_g", "_p")))
    return m, est


def response_functions(est):
    w = read_weights(str(ROOT / "runs/estimation/2026-08-27_11-01-35_global_IC_country_trends"
                              "/parameters/(2,).weights.h5"))
    K = np.asarray(w["layers/dense/vars/0"]); c0 = np.asarray(w["layers/dense/vars/1"])
    bt = np.asarray(w["layers/dense_1/vars/0"]).ravel()

    def f_nn(T, P):
        Tz, Pz = (T - MT) / ST, (P - MP) / SP
        o = 0.0
        for k in range(K.shape[1]):
            z = c0[k] + K[0, k] * Tz + K[1, k] * Pz
            o = o + bt[k] * (z / (1.0 + np.exp(-z)))
        return o

    e = est.copy()
    e["Pm"] = e.Pmm / 1000.0; e["T2"] = e["T"] ** 2; e["Pm2"] = e.Pm ** 2
    e["TP"] = e["T"] * e.Pm; e["T2P"] = e.T2 * e.Pm
    e["TP2"] = e["T"] * e.Pm2; e["T2P2"] = e.T2 * e.Pm2
    bB = within_fit(e, ["T", "T2", "Pm", "Pm2"])
    bL = within_fit(e, ["T", "T2", "Pm", "Pm2", "TP", "T2P", "TP2", "T2P2"])

    def f_quad(coef, poly):
        def f(T, P):
            Pm = P / 1000.0
            out = coef[0] * T + coef[1] * T ** 2
            if poly == "L":
                out = (out + coef[4] * T * Pm + coef[5] * T ** 2 * Pm
                       + coef[6] * T * Pm ** 2 + coef[7] * T ** 2 * Pm ** 2)
            return out
        return f

    return [("Neural network", f_nn),
            ("Burke (our sample)", f_quad(bB, "B")),
            ("Leirvik (our sample)", f_quad(bL, "L"))]


def paths(f, t0, p0, gdp0, ccd, G_, P_):
    """Burke's compounding loop -> per-country GDP/cap at 2099, with and without CC."""
    cc = gdp0.copy(); nc = gdp0.copy()
    bg = f(np.minimum(t0, TCAP), p0)
    for i in range(1, len(YRS)):
        nt = np.minimum(t0 + i * ccd, TCAP)
        gy = G_[:, i]
        nc = nc * (1.0 + gy)
        cc = cc * (1.0 + gy + f(nt, p0) - bg)
    return cc, nc


def main():
    m, est = build_inputs()
    t0 = m.Tours.to_numpy(float); p0 = m.Pmm.to_numpy(float)
    gdp0 = m.gdp.to_numpy(float); ccd = m.Tchg.to_numpy(float) / len(YRS)
    G_ = m[[f"{y}_g" for y in YRS]].to_numpy(float)
    P_ = m[[f"{y}_p" for y in YRS]].to_numpy(float)
    wt = P_[:, -1]                       # 2099 population, millions
    iso = m.CountryCode.to_numpy()
    nm = m.CNTRY_NAME.to_numpy()

    print(f"SSP5 / RCP 8.5, {len(m)} countries, impacts evaluated at 2099\n")
    rows = []

    for name, f in response_functions(est):
        cc, nc = paths(f, t0, p0, gdp0, ccd, G_, P_)
        imp = cc / nc - 1.0                                   # country impact
        contrib = wt * (cc - nc) / np.sum(wt * nc) * 100.0    # sums to the aggregate
        agg = contrib.sum()

        # ---------------- A. distribution vs the headline --------------------
        hurt_c = float((imp < 0).mean() * 100)
        hurt_p = float(wt[imp < 0].sum() / wt.sum() * 100)
        print(f"=== {name} ===")
        print(f"  A. headline aggregate (ratio of pop-weighted means) : {agg:+9.2f}%")
        print(f"     median country                                   : {np.median(imp)*100:+9.2f}%")
        print(f"     25th / 75th percentile country                   : "
              f"{np.percentile(imp,25)*100:+8.2f}% / {np.percentile(imp,75)*100:+.2f}%")
        print(f"     countries worse off                              : {hurt_c:9.1f}%")
        print(f"     world population worse off                       : {hurt_p:9.1f}%")

        # ---------------- B. who carries the aggregate -----------------------
        o = np.argsort(-contrib)
        print(f"\n  B. additive decomposition (contributions sum to {agg:+.2f}%)")
        print(f"     {'iso':>5} {'country':<22}{'pop 2099':>10}{'impact':>10}"
              f"{'GDPcap noCC':>13}{'GDPcap CC':>13}{'contrib':>10}")
        for k in o[:6]:
            print(f"     {iso[k]:>5} {str(nm[k])[:22]:<22}{wt[k]:9.1f}m{imp[k]*100:+9.0f}%"
                  f"{nc[k]:13,.0f}{cc[k]:13,.0f}{contrib[k]:+9.2f}%")
        print(f"     {'...':>5}")
        for k in o[-3:]:
            print(f"     {iso[k]:>5} {str(nm[k])[:22]:<22}{wt[k]:9.1f}m{imp[k]*100:+9.0f}%"
                  f"{nc[k]:13,.0f}{cc[k]:13,.0f}{contrib[k]:+9.2f}%")
        top1 = contrib[o[0]]
        pos = contrib[contrib > 0].sum()
        print(f"     largest single contributor        : {top1:+.2f}% "
              f"({iso[o[0]]}, {wt[o[0]]/wt.sum()*100:.2f}% of world population)")
        print(f"     all positive contributions        : {pos:+.2f}%")
        print(f"     all negative contributions        : {contrib[contrib<0].sum():+.2f}%")

        # ---------------- C. drop the largest winners ------------------------
        def agg_excl(drop):
            keep = np.ones(len(m), bool); keep[o[:drop]] = False
            return (np.sum(wt[keep]*cc[keep])/np.sum(wt[keep]*nc[keep]) - 1)*100
        print(f"\n  C. aggregate after removing the largest winners")
        print(f"     all countries {agg:+8.2f}%   drop top 1 {agg_excl(1):+8.2f}%   "
              f"drop top 3 {agg_excl(3):+8.2f}%   drop top 5 {agg_excl(5):+8.2f}%   "
              f"drop top 10 {agg_excl(10):+8.2f}%")

        # ---------------- D. aggregators compared ----------------------------
        pw_mean = float(np.average(imp, weights=wt) * 100)
        geo = float((np.exp(np.average(np.log(cc/nc), weights=wt)) - 1) * 100)
        unw = float(imp.mean() * 100)
        srt = np.argsort(imp); cw = np.cumsum(wt[srt]) / wt.sum()
        pw_med = float(imp[srt][np.searchsorted(cw, 0.5)] * 100)
        print(f"\n  D. same projection under five aggregators")
        print(f"     ratio of pop-weighted means (Burke headline) : {agg:+9.2f}%")
        print(f"     pop-weighted MEAN of country impacts         : {pw_mean:+9.2f}%")
        print(f"     pop-weighted MEDIAN country                  : {pw_med:+9.2f}%")
        print(f"     pop-weighted geometric mean                  : {geo:+9.2f}%")
        print(f"     unweighted mean of country impacts           : {unw:+9.2f}%\n")

        rows.append(dict(arm=name, aggregate=agg, median_country=np.median(imp)*100,
                         pw_mean=pw_mean, pw_median=pw_med, geometric=geo,
                         unweighted_mean=unw, countries_hurt=hurt_c, pop_hurt=hurt_p,
                         top1_iso=iso[o[0]], top1_contrib=top1,
                         drop_top1=agg_excl(1), drop_top5=agg_excl(5),
                         drop_top10=agg_excl(10)))

    df = pd.DataFrame(rows)
    dest = OUT / "2026-09-04-aggregation-decomposition.csv"
    df.to_csv(dest, index=False)
    print(f"table -> {dest}")


if __name__ == "__main__":
    main()
