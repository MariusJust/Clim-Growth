"""RCP 8.5 / SSP5 projections from the REGIONAL neural network, by region.

Each country is projected with its OWN region's response surface, evaluated in
that region's standardisation (see scripts/regional_surface.py -- the regional
network is trained on per-region z-scores, so using global moments evaluates
every head at the wrong point; that combination was verified against the
recorded BIC of the (2,) run to 0.025 out of -35,474.94).

Harness is Burke, Hsiang & Miguel (2015) ComputeMainProjections.R:
  ccd     = Tchg / 90                      (2010-2099)
  newtemp = temp + j*ccd,  j = 1..89       (temp = country mean 1980+)
  dg      = f(min(newtemp, 30)) - f(min(temp, 30))     response pinned at 30 C
  GDPcc_t = GDPcc_{t-1} * (1 + g_t + dg)   growth effects compound

Architecture: (2,), the BIC-best regional model (results.npy column 0).
Point estimates only -- there is no regional bootstrap run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from regional_surface import RegionalSurfaces, REGION_ORDER, REGION_CODE  # noqa: E402

YRS = list(range(2010, 2100))
TCAP = 30.0
NODE = (2,)
RUN = ROOT / "runs/estimation/2026-08-27_11-03-09_regional_IC_country_trends"
GLOBAL_W = ROOT / ("runs/estimation/2026-08-27_11-01-35_global_IC_country_trends"
                   "/parameters/(2,).weights.h5")
MT, ST, MP, SP = 20.3801, 7.2308, 1281.5723, 833.7378        # global-model moments
OUT = ROOT / "paper/Tables"
OUT.mkdir(parents=True, exist_ok=True)

_src = (ROOT / "scripts/2026-08-28-projections-corrected.py").read_text(encoding="utf-8")
_ns = {"__name__": "h", "__file__": str(ROOT / "scripts/2026-08-28-projections-corrected.py")}
exec(compile(_src.replace('if __name__ == "__main__":', 'if False:'), "h", "exec"), _ns)
ipolate, load_ours = _ns["ipolate"], _ns["load_ours"]


def build_inputs(mdata):
    """Same inputs as the global harness, plus each country's RegionCode."""
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

    reg = (mdata[["CountryCode", "RegionCode"]].drop_duplicates("CountryCode"))
    base = base.merge(reg, on="CountryCode", how="left")

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
    return m


def compound(dg, gdp0, G_):
    """Burke's loop. dg is (n_c, n_years); column t is the effect applied in year t."""
    cc = gdp0.copy(); nc = gdp0.copy()
    for i in range(1, len(YRS)):
        gy = G_[:, i]
        nc = nc * (1.0 + gy)
        cc = cc * (1.0 + gy + dg[:, i])
    return cc, nc


def main():
    mdata = pd.read_excel(ROOT / "data/MainData.xlsx")
    mdata = mdata[mdata.Year <= 2024]
    rs = RegionalSurfaces(RUN, NODE, mdata)

    m = build_inputs(mdata)
    code2reg = {v: k for k, v in REGION_CODE.items()}
    m["Region"] = m.RegionCode.map(code2reg)
    missing = m[m.Region.isna()]
    if len(missing):
        print(f"dropped {len(missing)} countries with no region: "
              f"{sorted(missing.CountryCode)}")
        m = m[m.Region.notna()].reset_index(drop=True)

    t0 = m.Tours.to_numpy(float); p0 = m.Pmm.to_numpy(float)
    gdp0 = m.gdp.to_numpy(float); ccd = m.Tchg.to_numpy(float) / len(YRS)
    G_ = m[[f"{y}_g" for y in YRS]].to_numpy(float)
    P_ = m[[f"{y}_p" for y in YRS]].to_numpy(float)
    wt = P_[:, -1]
    reg = m.Region.to_numpy()

    print(f"SSP5 / RCP 8.5, {len(m)} countries, regional NN {NODE}, "
          f"response pinned at {TCAP:.0f} C\n")
    print("  countries per region: " + ", ".join(
        f"{r} {int((reg == r).sum())}" for r in REGION_ORDER) + "\n")

    # ---- per-region optimum, in each region's own units ---------------------
    Tg = np.linspace(0, 30, 3001)
    print(f"  {'region':10}{'optimum T':>12}{'mean baseline T':>18}{'response range':>17}")
    for r in REGION_ORDER:
        sel = reg == r
        Pmed = float(np.median(p0[sel])) if sel.any() else float(np.median(p0))
        y = rs.f(r, Tg, Pmed)
        print(f"  {r:10}{Tg[int(np.argmax(y))]:11.2f} C"
              f"{t0[sel].mean() if sel.any() else np.nan:17.2f} C"
              f"{(y.max()-y.min())*100:16.2f}pp")

    # ---- regional NN impacts ------------------------------------------------
    dg = np.zeros((len(m), len(YRS)))
    for r in REGION_ORDER:
        sel = reg == r
        if not sel.any():
            continue
        tb = np.minimum(t0[sel], TCAP)
        base_r = rs.f(r, tb, p0[sel])
        path = np.minimum(t0[sel][:, None] + np.arange(len(YRS))[None, :] * ccd[sel][:, None],
                          TCAP)
        dg[sel] = rs.f(r, path, p0[sel][:, None]) - base_r[:, None]
    cc, nc = compound(dg, gdp0, G_)

    # ---- global model on the same countries, for comparison -----------------
    from hdf5_min import read_weights
    w = read_weights(str(GLOBAL_W))
    K = np.asarray(w["layers/dense/vars/0"]); c0 = np.asarray(w["layers/dense/vars/1"])
    bt = np.asarray(w["layers/dense_1/vars/0"]).ravel()

    def f_glob(T, P):
        Tz, Pz = (T - MT) / ST, (P - MP) / SP
        o = 0.0
        for j in range(K.shape[1]):
            z = c0[j] + K[0, j] * Tz + K[1, j] * Pz
            o = o + bt[j] * (z / (1.0 + np.exp(-z)))
        return o

    tb_g = np.minimum(t0, TCAP)
    path_g = np.minimum(t0[:, None] + np.arange(len(YRS))[None, :] * ccd[:, None], TCAP)
    dg_g = f_glob(path_g, p0[:, None]) - f_glob(tb_g, p0)[:, None]
    ccg, ncg = compound(dg_g, gdp0, G_)

    # ---- results ------------------------------------------------------------
    rows = []
    for lbl, a, b in (("Regional NN", cc, nc), ("Global NN", ccg, ncg)):
        imp = a / b - 1.0
        agg = (np.average(a, weights=wt) / np.average(b, weights=wt) - 1) * 100
        print(f"\n=== {lbl}: world, 2099 ===")
        print(f"  aggregate (ratio of pop-weighted means) : {agg:+9.2f}%")
        print(f"  median country                          : {np.median(imp)*100:+9.2f}%")
        print(f"  pop-weighted mean of country impacts    : "
              f"{np.average(imp, weights=wt)*100:+9.2f}%")
        print(f"  world population worse off              : "
              f"{wt[imp<0].sum()/wt.sum()*100:9.1f}%")
        print(f"\n  {'region':10}{'aggregate':>12}{'median ctry':>14}"
              f"{'pop hurt':>11}{'contrib to world':>19}")
        for r in REGION_ORDER:
            sel = reg == r
            if not sel.any():
                continue
            aR = (np.sum(wt[sel]*a[sel])/np.sum(wt[sel]*b[sel]) - 1)*100
            contrib = np.sum(wt[sel]*(a[sel]-b[sel]))/np.sum(wt*b)*100
            ph = wt[sel][imp[sel] < 0].sum()/wt[sel].sum()*100
            print(f"  {r:10}{aR:+11.2f}%{np.median(imp[sel])*100:+13.2f}%"
                  f"{ph:10.1f}%{contrib:+18.2f}%")
            rows.append(dict(model=lbl, region=r, n=int(sel.sum()),
                             aggregate=aR, median_country=np.median(imp[sel])*100,
                             pop_hurt=ph, contrib_to_world=contrib))
        rows.append(dict(model=lbl, region="WORLD", n=len(m), aggregate=agg,
                         median_country=np.median(imp)*100,
                         pop_hurt=wt[imp < 0].sum()/wt.sum()*100,
                         contrib_to_world=agg))

    dest = OUT / "2026-09-04-regional-projections.csv"
    pd.DataFrame(rows).to_csv(dest, index=False)
    print(f"\ntable -> {dest}")


if __name__ == "__main__":
    main()
