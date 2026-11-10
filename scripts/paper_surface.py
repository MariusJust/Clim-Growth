#!/usr/bin/env python3
"""Paper_1 climate response surfaces with the observed-density floor.

Reproduces the workflow in ``notebooks/global_model.ipynb`` -- select
architecture(s) from a run, rebuild the climate sub-network, evaluate it on the
(T, P) grid, and draw the Plotly surface with the 3-D density histogram on the
floor -- for the global, regional and income formulations, in the paper's exact
styling.

Runs WITHOUT TensorFlow: the climate sub-network is a small swish MLP, so the
forward pass is reconstructed in NumPy straight from the saved Keras weights.
That means figures regenerate anywhere (laptop, CI, sandbox), and it is the
same path that was verified to reproduce the published Figure 5 to <0.005.

Weight layout (derived structurally, so new runs work without edits)
-------------------------------------------------------------------
Layers are ``layers/dense*/vars/0`` (kernel) and ``vars/1`` (bias). Layers WITH
a bias are hidden layers; bias-free layers of width 1 are output heads. With R
output heads and Hn hidden layers the depth is H = Hn / R, and hidden layer h of
region r is index ``h*R + r`` -- i.e. the regional model carries a full
per-region stack, not a shared trunk. R = 1 recovers the global model.

Region order follows ``prepare_data.py``: Asia, Europe, Africa, Americas, Oceania.

Examples
--------
    # global, single BIC-best architecture (what the paper reports)
    python scripts/paper_surface.py runs/estimation/<run> --out results/images

    # global, mean of the 10 best (what the notebook does)
    python scripts/paper_surface.py runs/estimation/<run> --top 10

    # regional: one figure per region
    python scripts/paper_surface.py runs/estimation/<regional_run> --regional
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _go():
    """Import plotly lazily so the numeric core is usable without it."""
    import plotly.graph_objects as go
    return go

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from hdf5_min import read_weights          # scripts/hdf5_min.py, next to this file

# ---- paper styling (do not change without changing the paper) -------------
CIVIDIS = "Cividis"
CAMERA = dict(eye=dict(x=1.738, y=-1.780, z=0.589))
SURFACE_OPACITY = 0.85
BAR_MAX_HEIGHT = 0.12          # tallest density bar, in z units
BAR_OPACITY = 0.95

# Group order must match prepare_data.py, because output-head index == group index.
REGIONS = {"Asia": 142, "Europe": 150, "Africa": 2, "Americas": 19, "Oceania": 9}
INCOME_LABELS = {5: ["Q1", "Q2", "Q3", "Q4", "Q5"], 2: ["Low", "High"]}


# =========================================================== network rebuild
def _swish(z):
    return z / (1.0 + np.exp(-z))


def load_layers(weights_h5):
    """Split a saved model into hidden layers and output heads, in index order."""
    w = read_weights(str(weights_h5))
    idx = {}
    for key, arr in w.items():
        m = re.match(r"layers/(dense(?:_(\d+))?)/vars/(\d)$", key)
        if not m:
            continue
        i = int(m.group(2)) if m.group(2) else 0
        idx.setdefault(i, {})[int(m.group(3))] = np.asarray(arr)
    hidden, heads = [], []
    for i in sorted(idx):
        v = idx[i]
        if 1 in v:                                   # has bias -> hidden layer
            hidden.append((v[0], v[1]))
        else:                                        # bias-free -> output head
            heads.append(v[0])
    R = max(len(heads), 1)
    if len(hidden) % R:
        raise ValueError(f"{len(hidden)} hidden layers not divisible by {R} heads")
    return hidden, heads, R, len(hidden) // R


def centre_value(weights_h5, mean_T, std_T, mean_P, std_P, T0, P0, region_index=0):
    """Surface value at a reference climate (T0 degC, P0 mm) - the centring offset."""
    tz = np.array([[(T0 - mean_T) / std_T]]); pz = np.array([[(P0 - mean_P) / std_P]])
    return float(surface(weights_h5, tz, pz, region_index)[0, 0])


def surface(weights_h5, Tz, Pz, region_index=0):
    """Climate surface on standardised inputs, for one region (0 = global)."""
    hidden, heads, R, H = load_layers(weights_h5)
    X = np.column_stack([Tz.ravel(), Pz.ravel()])
    for h in range(H):
        K, b = hidden[h * R + region_index]
        X = _swish(X @ K + b)
    return (X @ heads[region_index]).reshape(Tz.shape)


# ================================================================ grid + data
def pred_grid(mean_T, std_T, mean_P, std_P, precip_capped=True, n=90):
    """Mirror of utils.create_pred_input(mc=False)."""
    t = np.linspace(0, 30, n)
    p = np.linspace(12.03731002, 3000 if precip_capped else 5435.30011, n)
    T, P = np.meshgrid(t, p)
    return T, P, (T - mean_T) / std_T, (P - mean_P) / std_P


def select_nodes(run_dir, criterion, top):
    """Rank architectures. results.npy values are [BIC, AIC] (+ holdout if present)."""
    res = dict(np.load(Path(run_dir) / "results.npy", allow_pickle=True).item())
    res = {k: v for k, v in res.items() if v is not None}
    col = {"BIC": 0, "AIC": 1, "holdout": 2}[criterion]
    ranked = sorted(res, key=lambda k: float(np.asarray(res[k], float).ravel()[col]))
    return ranked[:top], res


# ==================================================================== figure
def density_bins(t, p, precip_capped=True):
    """Edges + counts for the floor histogram (testable without plotly)."""
    t_edges = np.arange(0, int(np.ceil(t.max())) + 1, 1)                  # 1 C
    p_max = 3 if precip_capped else int(np.ceil(p.max()))
    p_edges = np.arange(int(np.floor(p.min())), p_max + 1, 1 / 3)         # 1/3 m
    counts, _, _ = np.histogram2d(t, p, bins=[t_edges, p_edges])
    return t_edges, p_edges, counts


def income_groups(data, n_groups, time_varying=True):
    """Assign each country-year to an income group, matching Prepare().

    With time_varying=True each country-year is ranked against the other
    countries observed in THAT year, so a country can change group over time.
    """
    labels = INCOME_LABELS.get(n_groups, [f"Q{i}" for i in range(1, n_groups + 1)])
    tmp = data[["CountryCode", "Year", "GDPCap"]].dropna(subset=["GDPCap"]).copy()
    if time_varying:
        cnt = tmp.groupby("Year")["GDPCap"].transform("count")
        tmp = tmp[cnt >= n_groups]
        grp = tmp.groupby("Year")["GDPCap"].transform(
            lambda s: pd.qcut(s.rank(method="first"), n_groups, labels=labels))
    else:                                    # fixed by long-run mean income
        mean_gdp = tmp.groupby("CountryCode")["GDPCap"].transform("mean")
        grp = pd.qcut(mean_gdp.rank(method="first"), n_groups, labels=labels)
    return grp.reindex(data.index), labels


def add_density_floor(fig, data, z_floor, region=None, income=None,
                      income_grp=None, precip_capped=True,
                      temp_col="TempPopWeight", precip_col="PrecipPopWeight",
                      showscale=False):
    """3-D histogram of observed (T, P) drawn on the floor (utils.add_histogram)."""
    go = _go()
    d = data
    if region is not None and income is not None:
        raise ValueError("specify either region or income, not both")
    if region is not None:
        code = REGIONS.get(region, region)
        d = d[d["RegionCode"] == code]
        if d.empty:
            raise ValueError(f"no observations for region {region!r}")
    if income is not None:
        if income_grp is None:
            raise ValueError("income filtering needs the precomputed group column")
        d = d[income_grp.to_numpy() == income]
        if d.empty:
            raise ValueError(f"no observations for income group {income!r}")
    t = pd.to_numeric(d[temp_col], errors="coerce").to_numpy(float)
    p = pd.to_numeric(d[precip_col], errors="coerce").to_numpy(float) / 1000.0
    ok = np.isfinite(t) & np.isfinite(p)
    t, p = t[ok], p[ok]

    t_edges, p_edges, counts = density_bins(t, p, precip_capped)
    cmax = counts.max()
    scaled = counts / cmax if cmax > 0 else counts

    for ix in range(len(t_edges) - 1):
        for iy in range(len(p_edges) - 1):
            if counts[ix, iy] <= 0:
                continue
            x0, x1 = t_edges[ix], t_edges[ix + 1]
            y0, y1 = p_edges[iy], p_edges[iy + 1]
            z1 = z_floor + scaled[ix, iy] * BAR_MAX_HEIGHT
            c = scaled[ix, iy]
            fig.add_trace(go.Mesh3d(
                x=[x0, x1, x1, x0, x0, x1, x1, x0],
                y=[y0, y0, y1, y1, y0, y0, y1, y1],
                z=[z_floor] * 4 + [z1] * 4,
                i=[0, 0, 0, 1, 4, 4, 3, 3, 0, 0, 1, 1],
                j=[1, 2, 4, 2, 5, 6, 2, 6, 3, 4, 2, 5],
                k=[2, 3, 5, 5, 6, 7, 6, 7, 4, 7, 6, 6],
                intensity=np.full(8, c), colorscale="YlOrRd", cmin=0, cmax=1,
                opacity=BAR_OPACITY, showscale=showscale, flatshading=True,
                hoverinfo="skip"))
    return fig


def make_figure(T, P, Z, data, zrange, region=None, income=None,
                income_grp=None, precip_capped=True,
                density=True, title=None, **hist_kw):
    go = _go()
    fig = go.Figure(go.Surface(x=T, y=P / 1000.0, z=Z, colorscale=CIVIDIS,
                               opacity=SURFACE_OPACITY, showscale=False,
                               name="surface"))
    fig.update_layout(
        autosize=True, margin=dict(l=0, r=0, b=0, t=0 if title is None else 30),
        title=title,
        scene=dict(xaxis_title="Temperature (°C)",
                   yaxis_title="Precipitation (m)",
                   zaxis=dict(title=dict(text="Δ ln(GDP)"), range=list(zrange)),
                   camera=CAMERA),
        font=dict(size=10))
    if density:
        add_density_floor(fig, data, zrange[0], region=region, income=income,
                          income_grp=income_grp,
                          precip_capped=precip_capped, **hist_kw)
    return fig


# ---- PDF export + the paper's exact crop ----------------------------------
# The published figures are a 1350 x 1050 pt Plotly/kaleido export cropped to
# 820.998 x 770.998 pt. The window was recovered by pixel-matching a cropped
# paper figure against its uncropped original (mean abs pixel error 0.000):
PDF_W, PDF_H = 1350, 1050
CROP = dict(left=210, right=319, top=198, bottom=81)      # points


def crop_pdf(path, crop=None, page_w=PDF_W, page_h=PDF_H):
    """Apply the paper's fixed crop window by rewriting the page boxes."""
    from pypdf import PdfReader, PdfWriter
    c = dict(CROP if crop is None else crop)
    r = PdfReader(str(path)); w = PdfWriter()
    for pg in r.pages:
        mb = pg.mediabox
        x0, y0 = float(mb.left), float(mb.bottom)
        # PDF origin is bottom-left; "top" is measured down from the top edge.
        pg.mediabox.lower_left = (x0 + c["left"], y0 + c["bottom"])
        pg.mediabox.upper_right = (x0 + page_w - c["right"], y0 + page_h - c["top"])
        pg.cropbox = pg.mediabox
        w.add_page(pg)
    with open(str(path), "wb") as fh:
        w.write(fh)
    return (page_w - c["left"] - c["right"], page_h - c["top"] - c["bottom"])


def write(fig, out_base, pdf=True, crop=True, scale=1):
    out_base = Path(out_base)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_base.with_suffix(".html")))
    if not pdf:
        return f"{out_base}.html"
    pdf_path = out_base.with_suffix(".pdf")
    try:
        fig.write_image(str(pdf_path), width=PDF_W, height=PDF_H, scale=scale)
    except Exception as e:
        return f"{out_base}.html  (PDF skipped -- install kaleido: {type(e).__name__})"
    if crop:
        try:
            w_, h_ = crop_pdf(pdf_path)
            return f"{out_base}.html + .pdf ({w_} x {h_} pt, paper crop)"
        except Exception as e:
            return f"{out_base}.html + .pdf (uncropped -- {type(e).__name__}: {e})"
    return f"{out_base}.html + .pdf ({PDF_W} x {PDF_H} pt, uncropped)"


# ====================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--data", default="data/MainData.xlsx")
    ap.add_argument("--out", default="results/images/surfaces")
    ap.add_argument("--criterion", default="BIC", choices=["BIC", "AIC", "holdout"])
    ap.add_argument("--top", type=int, default=1,
                    help="1 = single best (paper); 10 = mean of top 10 (notebook)")
    ap.add_argument("--node", default=None, help="force an architecture, e.g. '(2,)'")
    ap.add_argument("--regional", action="store_true", help="one figure per region")
    ap.add_argument("--income", action="store_true",
                    help="one figure per income group (labels inferred from the "
                         "number of output heads: 2 -> Low/High, 5 -> Q1..Q5)")
    ap.add_argument("--income-fixed", action="store_true",
                    help="assign income groups by long-run mean GDP instead of the "
                         "default time-varying within-year ranking")
    ap.add_argument("--zrange", type=float, nargs=2, default=(-0.3, 0.3))
    ap.add_argument("--no-capped", action="store_true",
                    help="extend precipitation axis to 5.435 m instead of 3 m")
    ap.add_argument("--no-density", action="store_true")
    ap.add_argument("--no-pdf", action="store_true", help="HTML only")
    ap.add_argument("--no-crop", action="store_true",
                    help="keep the full 1350x1050 export instead of the paper crop")
    ap.add_argument("--scale", type=float, default=1,
                    help="kaleido scale factor (raster detail; geometry unchanged)")
    ap.add_argument("--centre", default="auto",
                    choices=["auto", "none", "sample", "group"],
                    help="subtract the surface value at a reference climate. "
                         "'group' = that group's own observed mean (default for "
                         "--regional/--income, since each group's level is absorbed "
                         "by its own fixed effects); 'sample' = full-sample mean; "
                         "'auto' = group when grouped, else none")
    args = ap.parse_args()

    run = Path(args.run_dir)
    cfg_p = run / ".hydra/config.yaml"
    cfg = {}
    if cfg_p.exists():
        cfg = (yaml.safe_load(cfg_p.read_text()) or {}).get("instance", {})
    src = str((cfg.get("model") or cfg).get("data_source", "wb")).lower()

    data = (pd.read_excel(args.data) if str(args.data).endswith((".xlsx", ".xls"))
            else pd.read_csv(args.data, sep=";" if src == "ee" else ","))
    tcol = "temperature (celsius)" if src == "ee" else "TempPopWeight"
    pcol = "precipitation (mm)" if src == "ee" else "PrecipPopWeight"
    mT, sT = np.nanmean(data[tcol]), np.nanstd(data[tcol])
    mP, sP = np.nanmean(data[pcol]), np.nanstd(data[pcol])
    capped = not args.no_capped
    T, P, Tz, Pz = pred_grid(mT, sT, mP, sP, precip_capped=capped)
    print(f"data {len(data):,} rows | T ~ N({mT:.2f}, {sT:.2f}) | P ~ N({mP:.0f}, {sP:.0f})")

    if args.node:
        nodes = [args.node]
    else:
        nodes, res = select_nodes(run, args.criterion, args.top)
        print(f"{args.criterion}-ranked top {len(nodes)}: {nodes}")

    grouped = args.regional or args.income
    mode = args.centre
    if mode == "auto":
        mode = "group" if grouped else "none"

    # discover how many output heads this run has -> that is the group count
    wf0 = run / "parameters" / f"{nodes[0]}.weights.h5"
    _, _heads, R, H = load_layers(wf0)

    inc_grp = None
    if args.income:
        names = INCOME_LABELS.get(R, [f"Q{i}" for i in range(1, R + 1)])
        inc_grp, _ = income_groups(data, R, time_varying=not args.income_fixed)
        print(f"income model: {R} heads, depth {H} -> {names} "
              f"({'time-varying' if not args.income_fixed else 'fixed'} groups)")
    elif args.regional:
        names = list(REGIONS)[:R] if R <= len(REGIONS) else [f"g{i}" for i in range(R)]
        print(f"regional model: {R} heads, depth {H} -> {names}")
    else:
        names = [None]

    def subset_for(name):
        """Rows belonging to one group, for centring and for the density floor."""
        if name is None:
            return data
        if args.income:
            return data[inc_grp.to_numpy() == name]
        code = REGIONS.get(name, name)
        return data[data["RegionCode"] == code] if "RegionCode" in data.columns else data

    def surf_for(idx, name=None):
        zs = []
        for nd in nodes:
            wf = run / "parameters" / f"{nd}.weights.h5"
            if not wf.exists():
                print(f"   ! missing {wf.name}, skipped"); continue
            Z = surface(wf, Tz, Pz, idx)
            if mode != "none":
                sub = subset_for(name) if mode == "group" else data
                if len(sub) == 0:
                    sub = data
                T0 = float(np.nanmean(pd.to_numeric(sub[tcol], errors="coerce")))
                P0 = float(np.nanmean(pd.to_numeric(sub[pcol], errors="coerce")))
                Z = Z - centre_value(wf, mT, sT, mP, sP, T0, P0, idx)
            zs.append(Z)
        if not zs:
            raise SystemExit("no weight files found")
        return np.mean(zs, axis=0)

    for idx, name in enumerate(names):
        Z = surf_for(idx, name)
        fig = make_figure(
            T, P, Z, data, args.zrange,
            region=name if args.regional else None,
            income=name if args.income else None,
            income_grp=inc_grp,
            precip_capped=capped, density=not args.no_density,
            temp_col=tcol, precip_col=pcol)
        label = name if name else "global"
        print(f"  {label:10s} range [{Z.min():+.3f}, {Z.max():+.3f}]")
        print("  wrote", write(fig, f"{args.out}_{label}",
                               pdf=not args.no_pdf, crop=not args.no_crop,
                               scale=args.scale))


if __name__ == "__main__":
    main()
