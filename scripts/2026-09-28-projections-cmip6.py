"""SSP5-8.5 projections on the CMIP6 ensemble, 2024 to 2100.

Two uncertainty sources are reported: a 1,000-draw country-cluster bootstrap
(estimation) and the range across the eight CMIP6 models (climate). Regional
figures use the global response surface; regions are a reporting grouping, not a
separate estimate.

    python scripts/2026-09-28-projections-cmip6.py
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

from matplotlib.colors import ListedColormap, to_rgb                 # noqa: E402

BAND, LINE = "#2171b5", "black"

# Visually weighted band, after Hsiang (2012). Instead of stacking discrete
# quantile intervals, the density of the bootstrap trajectories is rendered
# directly: at each year the draws are binned over a vertical grid, smoothed, and
# normalised to that year's maximum, so colour saturation is the estimated
# likelihood that a trajectory passes through a given value CONDITIONAL on the
# year. Shading is clipped to the 2.5 to 97.5 percentile range, so the outer
# extent still reads as the 95 per cent interval.
# All six panels share one vertical scale, so the regions are directly
# comparable. None derives the range from the data; set a tuple such as
# (-100.0, 200.0) to clip the long Europe and Americas upper tails.
YLIM = (-100.0, 200.0)
DENS_N = 400          # vertical resolution of the density image
DENS_SMOOTH = 10.0    # Gaussian smoothing, in grid cells
DENS_ALPHA = 1.0      # opacity at the densest point
# Following Burke, Hsiang and Miguel (2015) Figure 4: ONE saturated hue at
# varying opacity, with plain black lines on top. The hue is fixed, so density is
# the only thing varying. Chroma is what makes this legible, not the hue itself:
# the first attempt used navy #00224e, which desaturates toward grey at low alpha
# and so sits close to the white background, while this mid-tone blue stays
# recognisably blue down to very low density. A light-to-dark "Blues" ramp was
# also tried and was worse, because lightness then varied alongside opacity.
DENS_COLOR = "#2171b5"
DENS_GAMMA = 0.99     # < 1 lifts mid densities, widening the visible range
_x = np.linspace(0.0, 1.0, 256)
_cols = np.zeros((256, 4))
_cols[:, :3] = to_rgb(DENS_COLOR)
_cols[:, 3] = (_x ** DENS_GAMMA) * DENS_ALPHA
DENS_CMAP = ListedColormap(_cols)
DENS_LEGEND = (*to_rgb(DENS_COLOR), 0.75)


def density_image(band, ygrid, lo_q=2.5, hi_q=97.5, smooth=DENS_SMOOTH):
    """(len(ygrid), n_years) array of per-year normalised trajectory density."""
    step = ygrid[1] - ygrid[0]
    edges = np.r_[ygrid - step / 2.0, ygrid[-1] + step / 2.0]
    half = max(int(3 * smooth), 1)
    k = np.exp(-0.5 * (np.arange(-half, half + 1) / smooth) ** 2)
    k /= k.sum()
    M = np.zeros((len(ygrid), band.shape[1]))
    for t in range(band.shape[1]):
        v = band[:, t]
        h = np.convolve(np.histogram(v, bins=edges)[0].astype(float), k, mode="same")
        lo, hi = np.percentile(v, [lo_q, hi_q])
        h[(ygrid < lo) | (ygrid > hi)] = 0.0
        if h.max() > 0:
            h /= h.max()
        M[:, t] = h
    return M
plt.rcParams.update({"font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13.5,
                     "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "legend.fontsize": 12})

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hdf5_min import read_weights                                   # noqa: E402

CMIP = ROOT / "data/projections/cmip6"
BOOT = ROOT / "reproduction/weights/2026-08-29_20-56-38_global_IC_bootstrap"
PEST = ROOT / ("reproduction/weights/2026-08-27_11-01-35_global_IC_country_trends"
               "/parameters/(2,).weights.h5")
OUT = ROOT / "paper/Figures/Projections"

YEARS = list(range(2024, 2101))
BASE0, BASE1 = 2015, 2024
TCAP = 30.0
MT, ST, MP, SP = 20.3801, 7.2308, 1281.5723, 833.7378
ISO_FIX = {"COD": "ZAR", "ROU": "ROM"}
REGIONS = {142: "Asia", 150: "Europe", 2: "Africa", 19: "Americas", 9: "Oceania"}
ORDER = ["Africa", "Asia", "Americas", "Europe", "Oceania"]
PANELS = ["Global", "Europe", "Africa", "Asia", "Americas", "Oceania"]
QB = ["T", "T2", "Pm", "Pm2"]
QL = ["T", "T2", "Pm", "Pm2", "TP", "T2P", "TP2", "T2P2"]


def save(fig, stem):
    for e in ("pdf", "png"):
        try:
            fig.savefig(OUT / f"{stem}.{e}", dpi=170, bbox_inches="tight")
        except PermissionError:
            print(f"  WARNING {stem}.{e} is locked (open in a viewer?), not written")


def swish(z):
    return z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


def nn_forward(w, X):
    idx = sorted({int(k.split("/")[1].split("_")[1]) if "_" in k.split("/")[1] else 0
                  for k in w if k.startswith("layers")})
    h = X
    for i in idx:
        b = f"layers/dense{'' if i == 0 else f'_{i}'}/vars"
        h = h @ np.asarray(w[f"{b}/0"], float)
        if f"{b}/1" in w:
            h = swish(h + np.asarray(w[f"{b}/1"], float))
    return h[..., 0]


def interp_weights(vals, grid):
    v = np.clip(vals, grid[0], grid[-1])
    i1 = np.clip(np.searchsorted(grid, v), 1, len(grid) - 1)
    i0 = i1 - 1
    w1 = (v - grid[i0]) / (grid[i1] - grid[i0])
    return i0, i1, 1.0 - w1, w1


def within_fit(df, cols):
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


def quad_coefs():
    d = pd.read_csv(ROOT / "data/MainData_est.csv", low_memory=False)
    e = pd.DataFrame(dict(CountryCode=d.CountryCode, Year=d.Year,
                          growth=pd.to_numeric(d.GrowthWDI, errors="coerce"),
                          T=pd.to_numeric(d.TempPopWeight, errors="coerce"),
                          Pm=pd.to_numeric(d.PrecipPopWeight, errors="coerce") / 1000.0)).dropna()
    e["T2"] = e["T"] ** 2; e["Pm2"] = e.Pm ** 2
    e["TP"] = e["T"] * e.Pm; e["T2P"] = e.T2 * e.Pm
    e["TP2"] = e["T"] * e.Pm2; e["T2P2"] = e.T2 * e.Pm2
    return within_fit(e, QB), within_fit(e, QL)


def ipolate(df, years):
    cols = {int(str(c)[:4]): c for c in df.columns if str(c)[:4].isdigit()}
    known = np.array(sorted(cols))
    out = np.empty((len(df), len(years)))
    for i, y in enumerate(years):
        if y in cols:
            out[:, i] = pd.to_numeric(df[cols[y]], errors="coerce")
            continue
        lo, hi = known[known < y], known[known > y]
        if not len(lo):
            out[:, i] = pd.to_numeric(df[cols[hi[0]]], errors="coerce")
        elif not len(hi):
            out[:, i] = pd.to_numeric(df[cols[lo[-1]]], errors="coerce")
        else:
            a, b = lo[-1], hi[0]
            va = pd.to_numeric(df[cols[a]], errors="coerce")
            vb = pd.to_numeric(df[cols[b]], errors="coerce")
            out[:, i] = va + (vb - va) * (y - a) / (b - a)
    return pd.DataFrame(out, columns=[str(y) for y in years]).assign(Region=df["Region"].values)


def load_panel():
    d = pd.read_excel(ROOT / "data/MainData.xlsx")
    d = d[d.Year <= BASE1]
    b = d[d.Year.between(BASE0, BASE1)]
    base = (b.groupby("CountryCode", as_index=False)
              .agg(T0=("TempPopWeight", "mean"), P0=("PrecipPopWeight", "mean")))
    gdp = (d.dropna(subset=["GDPCap"]).sort_values("Year")
             .groupby("CountryCode", as_index=False).GDPCap.last()
             .rename(columns={"GDPCap": "gdp0"}))
    reg = (d.dropna(subset=["RegionCode"]).groupby("CountryCode", as_index=False)
             .RegionCode.first())
    reg["region"] = reg.RegionCode.map(REGIONS)
    return base.merge(gdp, on="CountryCode").merge(
        reg[["CountryCode", "region"]], on="CountryCode").dropna()


# Each climate scenario must be paired with its OWN socioeconomic baseline.
# SSP5-8.5 reaches 8.5 W/m2 because it is a high-growth fossil world, so running
# SSP2-4.5 temperatures against SSP5 income and population describes no world at
# all and corrupts the population weighting in the aggregate.
SCENARIOS = {
    "ssp126": dict(label="SSP1-2.6", growth="SSP1_v9_130325", pop="SSP1_v9_130115"),
    "ssp245": dict(label="SSP2-4.5", growth="SSP2_v9_130325", pop="SSP2_v9_130115"),
    "ssp585": dict(label="SSP5-8.5", growth="SSP5_v9_130325", pop="SSP5_v9_130115"),
}


def load_climate(sid="ssp585"):
    e = pd.read_csv(CMIP / f"country_tas_ensemble_{sid}.csv")
    ens = e[e.year.isin(YEARS)].pivot(index="iso", columns="year", values="delta_C")
    m = pd.read_csv(CMIP / f"country_tas_by_model_{sid}.csv")
    m = m[m.year.isin(YEARS)]
    base = m[m.year.between(BASE0, BASE1)].groupby(["model", "iso"]).tas_C.mean()
    m = m.join(base.rename("b"), on=["model", "iso"])
    m["d"] = m.tas_C - m.b
    return ens, {k: g.pivot(index="iso", columns="year", values="d")
                 for k, g in m.groupby("model")}


def load_ssp(sid="ssp585"):
    spec = SCENARIOS[sid]
    g = pd.read_csv(ROOT / "data/projections/old/in/SSP/SSP_GrowthProjections.csv",
                    encoding="latin-1")
    p = pd.read_csv(ROOT / "data/projections/old/in/SSP/SSP_PopulationProjections.csv",
                    encoding="latin-1")
    g = g[(g.Model == "OECD Env-Growth") & (g.Scenario == spec["growth"])]
    p = p[p.Scenario == spec["pop"]]
    if g.empty or p.empty:
        raise SystemExit(f"{sid}: no SSP rows for {spec['growth']} / {spec['pop']}")
    ga, pa = ipolate(g, YEARS), ipolate(p, YEARS)
    ga[[str(y) for y in YEARS]] /= 100.0
    return ga.set_index("Region"), pa.set_index("Region")


def response_nn(T, P):
    w = read_weights(str(PEST))
    X = np.stack([(T - MT) / ST, np.broadcast_to((P - MP) / SP, T.shape)], axis=-1)
    return nn_forward(w, X)


def quad_dg(coef, kind, Tpath, Tbase, P0):
    c = np.atleast_2d(coef)
    Pm = (P0 / 1000.0)[None, :, None]
    dT = (Tpath - Tbase[:, None])[None]
    dT2 = (Tpath ** 2 - (Tbase ** 2)[:, None])[None]
    out = c[:, 0][:, None, None] * dT + c[:, 1][:, None, None] * dT2
    if kind == "L":
        out = (out + c[:, 4][:, None, None] * dT * Pm
                   + c[:, 5][:, None, None] * dT2 * Pm
                   + c[:, 6][:, None, None] * dT * Pm ** 2
                   + c[:, 7][:, None, None] * dT2 * Pm ** 2)
    return out


def project(dg, gdp0, G, W, masks):
    R, n, K = dg.shape
    cc = np.tile(gdp0, (R, 1)); nc = gdp0.copy()
    agg = np.zeros((R, K)); med = np.zeros((R, K)); hrm = np.zeros((R, K))
    reg = {r: np.zeros((R, K)) for r in masks}
    for i in range(1, K):
        gy = G[:, i]
        nc = nc * (1.0 + gy); cc = cc * (1.0 + gy[None, :] + dg[:, :, i])
        wt = W[:, i]
        imp = cc / nc[None, :] - 1.0
        agg[:, i] = (np.average(cc, weights=wt, axis=1)
                     / np.average(nc, weights=wt) - 1.0) * 100
        med[:, i] = np.median(imp, axis=1) * 100
        hrm[:, i] = (imp < 0) @ wt / wt.sum() * 100
        for r, m in masks.items():
            reg[r][:, i] = (np.average(cc[:, m], weights=wt[m], axis=1)
                            / np.average(nc[m], weights=wt[m]) - 1.0) * 100
    return dict(agg=agg, med=med, hrm=hrm, reg=reg, cc=cc, nc=nc)


def run_scenario(sid):
    """Everything for one SSP. Returns the pieces the cross-scenario figure needs."""
    spec = SCENARIOS[sid]
    tag = "" if sid == "ssp585" else f"_{sid}"
    print(f"\n{'='*72}\n{spec['label']}  ({sid})\n{'='*72}")

    panel = load_panel()
    ens, per_model = load_climate(sid)
    panel = panel[panel.CountryCode.isin(ens.index)].reset_index(drop=True)
    panel["ssp"] = panel.CountryCode.replace(ISO_FIX)
    G_df, P_df = load_ssp(sid)
    panel = panel[panel.ssp.isin(G_df.index) & panel.ssp.isin(P_df.index)].reset_index(drop=True)

    iso = panel.CountryCode.to_numpy()
    T0, P0 = panel.T0.to_numpy(float), panel.P0.to_numpy(float)
    gdp0 = panel.gdp0.to_numpy(float)
    G = G_df.loc[panel.ssp, [str(y) for y in YEARS]].to_numpy(float)
    W = P_df.loc[panel.ssp, [str(y) for y in YEARS]].to_numpy(float)
    masks = {r: (panel.region == r).to_numpy() for r in ORDER
             if (panel.region == r).any()}

    Tbase = np.minimum(T0, TCAP)
    D = ens.loc[iso, YEARS].to_numpy(float)
    paths = {"ensemble": np.minimum(T0[:, None] + D, TCAP)}
    for nm, dfm in per_model.items():
        paths[nm] = np.minimum(T0[:, None] + dfm.loc[iso, YEARS].to_numpy(float), TCAP)
    models = [k for k in paths if k != "ensemble"]
    print(f"countries {len(iso)}   years {YEARS[0]}-{YEARS[-1]} ({len(YEARS)-1} steps)   "
          f"models {len(models)}   warming 2100 median {np.median(D[:, -1]):.2f} C")

    bB, bL = quad_coefs()
    f0 = response_nn(Tbase, P0)
    nn_dg = lambda T: (response_nn(T, P0[:, None]) - f0[:, None])[None]
    arms = {"Network": nn_dg,
            "Burke": lambda T: quad_dg(bB, "B", T, Tbase, P0),
            "Leirvik": lambda T: quad_dg(bL, "L", T, Tbase, P0)}
    res = {a: {p: project(f(paths[p]), gdp0, G, W, masks) for p in paths}
           for a, f in arms.items()}

    S = np.load(BOOT / "bootstrap_surfaces.npy")[:, 0].astype(float)
    zz = np.load(BOOT / "bootstrap_bands.npz", allow_pickle=True)
    Tg, Pg = zz["T"][0, :], zz["P"][:, 0]
    pj0, pj1, pw0, pw1 = interp_weights(P0, Pg)
    rows = S[:, pj0, :] * pw0[None, :, None] + S[:, pj1, :] * pw1[None, :, None]
    cidx = np.arange(len(iso))[:, None]

    def on_T(T):
        i0, i1, w0, w1 = interp_weights(T, Tg)
        if T.ndim == 1:
            return rows[:, np.arange(len(iso)), i0] * w0 + rows[:, np.arange(len(iso)), i1] * w1
        return rows[:, cidx, i0] * w0 + rows[:, cidx, i1] * w1

    Tp = paths["ensemble"]
    bs = {"Network": project(on_T(Tp) - on_T(Tbase)[:, :, None], gdp0, G, W, masks)}
    # Paired draws: the quadratic is refitted on the SAME country resamples as the
    # network, so Network - Burke can be formed draw by draw. The unpaired file is
    # only good for marginal bands.
    qp = ROOT / "data/bootstrap_quad_coefs_paired.npz"
    z = np.load(qp if qp.exists() else ROOT / "data/bootstrap_quad_coefs.npz")
    PAIRED = qp.exists()
    for nm, kind, key in (("Burke", "B", "burke"), ("Leirvik", "L", "leirvik")):
        dr = z[key][~np.isnan(z[key]).any(1)]
        bs[nm] = project(quad_dg(dr, kind, Tp, Tbase, P0), gdp0, G, W, masks)
    print(f"bootstrap draws: NN {S.shape[0]}, quadratics {z['burke'].shape[0]}")

    def cells(arm, key, reg=None):
        pt = (res[arm]["ensemble"]["reg"][reg] if reg else res[arm]["ensemble"][key])[0, -1]
        bb = (bs[arm]["reg"][reg] if reg else bs[arm][key])[:, -1]
        mv = [(res[arm][m]["reg"][reg] if reg else res[arm][m][key])[0, -1] for m in models]
        lo, hi = np.percentile(bb, [2.5, 97.5])
        return pt, (lo, hi), (min(mv), max(mv))

    print(f"\n=== {YEARS[-1]} change in GDP per capita, {spec['label']} ===")
    print(f"{'':24}{'estimate':>10}{'95% bootstrap':>22}{'across models':>20}")
    for arm in arms:
        print(f"  {arm}")
        for lab, key in (("median country", "med"), ("population worse off", "hrm"),
                         ("aggregate", "agg")):
            pt, b, m = cells(arm, key)
            print(f"    {lab:20}{pt:>10.1f}   [{b[0]:>7.1f}, {b[1]:>7.1f}]"
                  f"   [{m[0]:>6.1f}, {m[1]:>6.1f}]")

    print(f"\n=== {YEARS[-1]} regional aggregates (Network, global surface) ===")
    print(f"{'region':10}{'N':>4}{'base T':>8}{'warming':>9}{'aggregate':>11}"
          f"{'95% bootstrap':>22}{'across models':>20}")
    rows_out = []
    for r, m in masks.items():
        pt, b, mr = cells("Network", None, reg=r)
        print(f"  {r:10}{m.sum():>3}{T0[m].mean():>8.1f}{D[m,-1].mean():>8.2f}"
              f"{pt:>11.1f}   [{b[0]:>7.1f}, {b[1]:>7.1f}]   [{mr[0]:>6.1f}, {mr[1]:>6.1f}]")
        rows_out.append(dict(region=r, n=int(m.sum()), base_T=T0[m].mean(),
                             warming=D[m, -1].mean(), aggregate=pt,
                             boot_lo=b[0], boot_hi=b[1],
                             model_lo=mr[0], model_hi=mr[1]))

    # Per-country impacts and contributions to the aggregate. The aggregate is a
    # ratio of means, so it decomposes as A = sum_i p_i (Yc_i - Y0_i) / sum_j p_j Y0_j:
    # a country's contribution weights its percentage impact by population times
    # baseline output, which is why the two rankings differ.
    wt = W[:, -1]
    rn = res["Network"]["ensemble"]
    cc, nc = rn["cc"][0], rn["nc"]
    impact = (cc / nc - 1.0) * 100
    contrib = wt * (cc - nc) / np.sum(wt * nc) * 100

    def leave(arm, k):
        r = res[arm]["ensemble"]
        c, n = r["cc"][0], r["nc"]
        o = np.argsort(-(wt * (c - n)))[k:]
        return (np.average(c[o], weights=wt[o]) / np.average(n[o], weights=wt[o]) - 1) * 100

    oi = np.argsort(-impact)
    oc = np.argsort(-contrib)
    print(f"\n=== {YEARS[-1]} largest country impacts (Network) ===")
    print(f"{'':4}{'gains':>28}{'':6}{'losses':>28}")
    print(f"  {'iso':>4}{'baseT':>7}{'warm':>7}{'impact':>10}"
          f"    {'iso':>4}{'baseT':>7}{'warm':>7}{'impact':>10}")
    for a, b_ in zip(oi[:10], oi[::-1][:10]):
        print(f"  {iso[a]:>4}{T0[a]:>7.1f}{D[a,-1]:>7.2f}{impact[a]:>9.1f}%"
              f"    {iso[b_]:>4}{T0[b_]:>7.1f}{D[b_,-1]:>7.2f}{impact[b_]:>9.1f}%")

    print(f"\n=== {YEARS[-1]} largest contributions to the aggregate (pp) ===")
    print(f"  {'iso':>4}{'contrib':>9}{'impact':>10}    {'iso':>4}{'contrib':>9}{'impact':>10}")
    for a, b_ in zip(oc[:8], oc[::-1][:8]):
        print(f"  {iso[a]:>4}{contrib[a]:>+9.2f}{impact[a]:>9.1f}%"
              f"    {iso[b_]:>4}{contrib[b_]:>+9.2f}{impact[b_]:>9.1f}%")
    print(f"  sum of contributions {contrib.sum():+.2f} pp "
          f"(aggregate {rn['agg'][0, -1]:+.2f}%)")

    print(f"\n=== {YEARS[-1]} aggregate excluding the largest contributors ===")
    print(f"{'':22}{'Network':>10}{'Burke':>10}{'Leirvik':>10}")
    for k in (3, 5):
        print(f"  {'excl. top ' + str(k):20}" + "".join(f"{leave(a, k):>10.1f}" for a in arms))

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(iso=iso, region=panel.region.to_numpy(), base_T=T0,
                      warming=D[:, -1], impact_pct=impact, contribution_pp=contrib)
                 ).sort_values("impact_pct", ascending=False).to_csv(
        OUT / f"projections_cmip6_country_impacts_2100{tag}.csv", index=False)
    pd.DataFrame(rows_out).to_csv(OUT / f"projections_cmip6_regions_2100{tag}.csv", index=False)
    pd.DataFrame({"year": YEARS,
                  **{r: res["Network"]["ensemble"]["reg"][r][0] for r in masks},
                  **{f"{r}_boot_lo": np.percentile(bs["Network"]["reg"][r], 2.5, axis=0)
                     for r in masks},
                  **{f"{r}_boot_hi": np.percentile(bs["Network"]["reg"][r], 97.5, axis=0)
                     for r in masks},
                  **{f"{r}_mod_lo": np.min([res["Network"][m]["reg"][r][0] for m in models], 0)
                     for r in masks},
                  **{f"{r}_mod_hi": np.max([res["Network"][m]["reg"][r][0] for m in models], 0)
                     for r in masks}}).to_csv(
        OUT / f"projections_cmip6_regions{tag}.csv", index=False)

    def series(name):
        if name == "Global":
            return (res["Network"]["ensemble"]["agg"][0], bs["Network"]["agg"],
                    len(iso), T0.mean())
        m = masks[name]
        return (res["Network"]["ensemble"]["reg"][name][0], bs["Network"]["reg"][name],
                int(m.sum()), T0[m].mean())

    glob_out = dict(agg=res["Network"]["ensemble"]["agg"][0],
                    agg_draws=bs["Network"]["agg"],
                    med=res["Network"]["ensemble"]["med"][0],
                    med_draws=bs["Network"]["med"],
                    burke_agg=res["Burke"]["ensemble"]["agg"][0],
                    burke_agg_draws=bs["Burke"]["agg"],
                    paired=PAIRED)
    if sid != "ssp585":
        print(f"\n  -> {OUT/('projections_cmip6_regions_2100' + tag + '.csv')}")
        return glob_out

    drawn = {name: series(name) for name in PANELS}
    if YLIM is None:
        ylo = min(min(np.percentile(b, 2.5, axis=0).min(), p.min())
                  for p, b, _, _ in drawn.values())
        yhi = max(max(np.percentile(b, 97.5, axis=0).max(), p.max())
                  for p, b, _, _ in drawn.values())
    else:
        ylo, yhi = YLIM
    pad = 0.0 if YLIM is not None else 0.06 * (yhi - ylo)
    ygrid = np.linspace(ylo - pad, yhi + pad, DENS_N)

    fig, axes = plt.subplots(2, 3, figsize=(14.0, 8.0), sharex=True, sharey=True)
    for j, (ax, name) in enumerate(zip(axes.ravel(), PANELS)):
        pt, band, n, tb = drawn[name]
        ax.fill_between(YEARS, np.percentile(band, 2.5, axis=0),
                        np.percentile(band, 97.5, axis=0),
                        color=BAND, alpha=0.18, lw=0, zorder=1)
        ax.set_ylim(ygrid[0], ygrid[-1])
        ax.plot(YEARS, pt, color=LINE, lw=1.6, zorder=3)
        ax.axhline(0, color=LINE, lw=0.9, zorder=2)
        ax.set_title(f"{'Global model' if name == 'Global' else name}")
        if j >= 3:
            ax.set_xlabel("Year")
        if j % 3 == 0:
            ax.set_ylabel("% change in GDP per capita")
        if j == 0:
            # imshow returns an AxesImage, which matplotlib will not place in a
            # legend, so the density entry needs a proxy patch.
            from matplotlib.patches import Patch
            from matplotlib.lines import Line2D
            # ax.legend(handles=[Line2D([], [], color=LINE, lw=1.6,
            #                           label="point estimate"),
            #                    Patch(facecolor=DENS_LEGEND, edgecolor="none",
            #                          label="bootstrap density, 95% interval")],
            #           frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout()
    save(fig, "projections_cmip6")

    print(f"\n  -> {OUT/'projections_cmip6.pdf'}")
    print(f"  -> {OUT/'projections_cmip6_regions_2100.csv'}")
    print(f"  -> {OUT/'projections_cmip6_regions.csv'}")
    return glob_out


SCEN_COLOR = {"ssp126": "#238b45", "ssp245": "#f16913", "ssp585": "#cb181d"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenarios", nargs="*", default=list(SCENARIOS),
                    choices=list(SCENARIOS))
    a = ap.parse_args()

    out = {sid: run_scenario(sid) for sid in a.scenarios}

    # Global aggregate across scenarios. Regional detail stays on the SSP5-8.5
    # 2x3 figure; the scenario spread is a global story, and three density
    # fields per panel would be unreadable.
    fig, ax = plt.subplots(figsize=(8.0, 5.2))
    for sid in (a.scenarios if len(out) > 1 else []):
        pt, band = out[sid]["agg"], out[sid]["agg_draws"]
        c = SCEN_COLOR[sid]
        lo = np.percentile(band, 2.5, axis=0)
        hi = np.percentile(band, 97.5, axis=0)
        ax.fill_between(YEARS, lo, hi, color=c, alpha=0.13, lw=0)
        ax.plot(YEARS, pt, color=c, lw=2.0, label=SCENARIOS[sid]["label"])
    ax.axhline(0, color="0.4", lw=0.9)
    ax.set_xlabel("Year")
    ax.set_ylabel("% change in GDP per capita")
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    save(fig, "projections_cmip6_scenarios")

    print(f"\n=== {YEARS[-1]} global aggregate by scenario ===")
    print(f"{'scenario':12}{'estimate':>10}{'95% bootstrap':>24}")
    for sid in (a.scenarios if len(out) > 1 else []):
        pt, band = out[sid]["agg"], out[sid]["agg_draws"]
        lo, hi = np.percentile(band[:, -1], [2.5, 97.5])
        print(f"  {SCENARIOS[sid]['label']:10}{pt[-1]:>10.1f}   [{lo:>7.1f}, {hi:>7.1f}]")
    print(f"\n  -> {OUT/'projections_cmip6_scenarios.pdf'}")

    # ------------------------------------------------- median-country histogram
    # The distribution of the 2100 median country across bootstrap draws, one
    # histogram per scenario. Replaces reading uncertainty off a colour ramp.
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    allv = np.concatenate([out[s_]["med_draws"][:, -1] for s_ in a.scenarios])
    bins = np.linspace(np.percentile(allv, 0.2), np.percentile(allv, 99.8), 41)
    for sid in a.scenarios:
        v = out[sid]["med_draws"][:, -1]
        c = SCEN_COLOR[sid]
        ax.hist(v, bins=bins, histtype="step", lw=1.8, color=c,
                label=f"{SCENARIOS[sid]['label']}  (median {np.median(v):+.0f}%)")
        ax.axvline(out[sid]["med"][-1], color=c, lw=0.9, ls=":")
    ax.axvline(0, color="0.4", lw=0.9)
    ax.set_xlabel(f"% change in GDP per capita of the median country, {YEARS[-1]}")
    ax.set_ylabel("bootstrap draws")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "projections_cmip6_median_hist")

    print(f"\n=== {YEARS[-1]} median country, distribution across draws ===")
    print(f"{'scenario':12}{'point':>8}{'p2.5':>8}{'p50':>8}{'p97.5':>8}{'P(<0)':>8}")
    for sid in a.scenarios:
        v = out[sid]["med_draws"][:, -1]
        q = np.percentile(v, [2.5, 50, 97.5])
        print(f"  {SCENARIOS[sid]['label']:10}{out[sid]['med'][-1]:>8.1f}"
              f"{q[0]:>8.1f}{q[1]:>8.1f}{q[2]:>8.1f}{(v < 0).mean()*100:>7.0f}%")
    print(f"  -> {OUT/'projections_cmip6_median_hist.pdf'}")

    # ------------------------------------------- network against the quadratic
    sid = "ssp585" if "ssp585" in a.scenarios else a.scenarios[-1]
    o = out[sid]
    nn, bk = o["agg_draws"], o["burke_agg_draws"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.0, 7.0), sharex=True,
                                   gridspec_kw=dict(height_ratios=[2, 1]))
    for pt, dr, c, lab in ((o["agg"], nn, "#2171b5", "Network"),
                           (o["burke_agg"], bk, "#cb181d", "Burke quadratic")):
        ax1.fill_between(YEARS, np.percentile(dr, 2.5, axis=0),
                         np.percentile(dr, 97.5, axis=0), color=c, alpha=0.13, lw=0)
        ax1.plot(YEARS, pt, color=c, lw=2.0, label=lab)
    ax1.axhline(0, color="0.4", lw=0.9)
    ax1.set_ylabel("% change in GDP per capita")
    ax1.set_title(f"{SCENARIOS[sid]['label']}, population-weighted aggregate", loc="left")
    ax1.legend(frameon=False, loc="upper left")

    n = min(len(nn), len(bk))
    d = nn[:n] - bk[:n]
    ax2.fill_between(YEARS, np.percentile(d, 2.5, axis=0),
                     np.percentile(d, 97.5, axis=0), color="0.45", alpha=0.18, lw=0)
    ax2.plot(YEARS, o["agg"] - o["burke_agg"], color="black", lw=1.8)
    ax2.axhline(0, color="0.4", lw=0.9)
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Network - Burke (pp)")
    note = "paired draws" if o["paired"] else "UNPAIRED draws, difference not valid"
    ax2.set_title(note, loc="left", fontsize=10)
    fig.tight_layout()
    save(fig, "projections_cmip6_nn_vs_burke")

    lo, hi = np.percentile(d[:, -1], [2.5, 97.5])
    print(f"\n=== {YEARS[-1]} Network minus Burke, aggregate, {SCENARIOS[sid]['label']} ===")
    print(f"  point {o['agg'][-1] - o['burke_agg'][-1]:+.1f} pp"
          f"   95% [{lo:+.1f}, {hi:+.1f}]   P(>0) {(d[:, -1] > 0).mean()*100:.0f}%"
          f"   {'paired' if o['paired'] else 'UNPAIRED'}")
    print(f"  -> {OUT/'projections_cmip6_nn_vs_burke.pdf'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
