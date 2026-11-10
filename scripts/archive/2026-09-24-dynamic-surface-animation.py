"""Animated (T, P) response surface for the dynamic model, one frame per year.

    python scripts/2026-09-24-dynamic-surface-animation.py
    python scripts/2026-09-24-dynamic-surface-animation.py --run <dir> --node "(2, 2, 2)"

Produces a self-contained HTML file: press Play to sweep the years, or drag the
slider to hold any single year. Each frame shows that year's fitted surface plus
the country observations that identify it.

Observations are PARTIALLED OUT, not raw
----------------------------------------
The network is fitted to growth after the nuisance structure is projected out, so
raw growth is not on the same scale as the surface and overlaying it would be
misleading. For each year we therefore plot

    y_it - (W gamma)_it

where W holds the country fixed effects (plus year effects and country trends when
the run carries them) and gamma is recovered by exact OLS given the fitted climate
net, exactly as the training loss does it. That residualised value is what the
surface is trying to explain, so the two are directly comparable.

The time input is standardised with the same deterministic moments the training code
uses when time_scaling='standardize': mu = (T+1)/2, sigma = sqrt((T^2-1)/12). Feeding
raw year indices would evaluate the network far outside its training range.
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hdf5_min import read_weights  # noqa: E402

DEF_RUN = "runs/estimation/2026-09-08_12-44-40_global_dynamic"
DEF_NODE = "(2, 2, 2)"
CAMERA = dict(eye=dict(x=2.11, y=0.12, z=0.38))


def _within_projector():
    """Load WithinProjector by path: the package __init__ chain imports tensorflow."""
    spec = importlib.util.spec_from_file_location(
        "within_projection",
        ROOT / "src/models/helper_functions/global_model/within_projection.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.WithinProjector


def swish(z):
    return z / (1.0 + np.exp(-z))


def forward(w, X):
    """Chained Dense stack: hidden layers carry a bias, the output layer does not."""
    idx = sorted({int(k.split("/")[1].split("_")[1]) if "_" in k.split("/")[1] else 0
                  for k in w if k.startswith("layers")})
    h = X
    for i in idx:
        b = f"layers/dense{'' if i == 0 else f'_{i}'}/vars"
        h = h @ np.asarray(w[f"{b}/0"], float)
        if f"{b}/1" in w:
            h = swish(h + np.asarray(w[f"{b}/1"], float))
    return h[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEF_RUN)
    ap.add_argument("--node", default=DEF_NODE)
    ap.add_argument("--grid", type=int, default=70, help="grid points per axis")
    ap.add_argument("--outfile", default=None)
    ap.add_argument("--cdn", action="store_true",
                    help="load plotly.js from a CDN instead of embedding it; "
                         "gives a file a few MB smaller that needs an internet "
                         "connection to render")
    a = ap.parse_args()

    run = ROOT / a.run
    cfg = yaml.safe_load((run / ".hydra/config.yaml").read_text(encoding="utf-8"))["instance"]
    m = cfg["model"]
    dyn = bool(m.get("dynamic_model", False))
    ct = bool(m.get("country_trends", False))
    levels = str(m.get("target_mode", "growth")).lower() == "levels"

    data = pd.read_excel(ROOT / "data/MainData.xlsx")
    if m.get("data_end"):
        data = data[data.Year <= m["data_end"]]
    Tp = data.pivot(index="Year", columns="CountryCode", values="TempPopWeight")
    Pp = data.pivot(index="Year", columns="CountryCode", values="PrecipPopWeight").reindex_like(Tp)
    Yp = data.pivot(index="Year", columns="CountryCode",
                    values="GDPCap" if levels else "GrowthWDI").reindex_like(Tp)
    Tv, Pv = Tp.to_numpy(float), Pp.to_numpy(float)
    Yv = np.log(Yp.to_numpy(float)) if levels else Yp.to_numpy(float)
    years = np.asarray(Tp.index, dtype=int)
    nT, nN = Tv.shape

    mT, sT = float(np.nanmean(Tv)), float(np.nanstd(Tv))
    mP, sP = float(np.nanmean(Pv)), float(np.nanstd(Pv))
    mt, st = (nT + 1) / 2, np.sqrt((nT ** 2 - 1) / 12)

    w = read_weights(str(run / "parameters" / f"{a.node}.weights.h5"))

    # ---- observations, partialled out -------------------------------------
    WP = _within_projector()
    mask = np.isnan(Tv)
    pr = WP(mask, country_trends=ct, quadratic_trends=ct, include_time=not dyn)
    cols = [np.nan_to_num((Tv - mT) / sT), np.nan_to_num((Pv - mP) / sP)]
    if dyn:
        cols.append((np.arange(1, nT + 1)[:, None] - mt) / st * np.ones((1, nN)))
    f_full = forward(w, np.stack(cols, axis=-1))
    y_obs, f_obs = Yv[pr.t_arr, pr.n_arr], f_full[pr.t_arr, pr.n_arr]
    ok = np.isfinite(y_obs) & np.isfinite(f_obs)
    gamma = pr.recover_gamma(np.where(ok, y_obs - f_obs, 0.0))
    resid_obs = np.where(ok, y_obs, np.nan) - pr.W @ gamma      # comparable to the surface

    obs = pd.DataFrame(dict(year=years[pr.t_arr],
                            T=Tv[pr.t_arr, pr.n_arr],
                            P=Pv[pr.t_arr, pr.n_arr] / 1000.0,
                            z=resid_obs)).dropna()

    # ---- surface grid ------------------------------------------------------
    Tg = np.linspace(0.0, 30.0, a.grid)
    Pg = np.linspace(float(np.nanmin(Pv)), 3000.0, a.grid)      # precipitation capped at 3 m
    TT, PP = np.meshgrid(Tg, Pg)

    def surf_for(j):
        cs = [(TT - mT) / sT, (PP - mP) / sP]
        if dyn:
            cs.append(np.full_like(TT, ((j + 1) - mt) / st))
        return forward(w, np.stack(cs, axis=-1))

    Z = [surf_for(j) for j in range(nT)]
    zlo = min(float(np.min(z)) for z in Z)
    zhi = max(float(np.max(z)) for z in Z)
    zlo = min(zlo, float(obs.z.quantile(0.01)))
    zhi = max(zhi, float(obs.z.quantile(0.99)))
    pad = 0.05 * (zhi - zlo)

    print(f"{run.name}  node {a.node}  {nT} frames  grid {a.grid}x{a.grid}")
    print(f"  surface z range {min(np.min(z) for z in Z):.4f} .. {max(np.max(z) for z in Z):.4f}")
    print(f"  observations    {len(obs)} points, {obs.year.nunique()} years")

    def frame_traces(j):
        o = obs[obs.year == years[j]]
        return [
            go.Surface(x=TT, y=PP / 1000.0, z=Z[j], colorscale="Cividis",
                       cmin=zlo, cmax=zhi, opacity=0.9, showscale=False,
                       hoverinfo="skip", name="Model"),
            go.Scatter3d(x=o["T"], y=o["P"], z=o["z"], mode="markers",
                         name="Observations", showlegend=False,
                         marker=dict(size=4, opacity=0.9, color="#b23a48",
                                     line=dict(width=0.3)),
                         hovertemplate="%{text}<br>T %{x:.1f} C<br>P %{y:.2f} m"
                                       "<br>resid %{z:.3f}<extra></extra>",
                         text=list(o.index.map(lambda _: str(years[j])))),
        ]

    fig = go.Figure(
        data=frame_traces(0),
        frames=[go.Frame(data=frame_traces(j), name=str(years[j])) for j in range(nT)],
    )

    steps = [dict(label=str(y), method="animate",
                  args=[[str(y)], dict(mode="immediate",
                                       frame=dict(duration=200, redraw=True),
                                       transition=dict(duration=0))])
             for y in years]

    fig.update_layout(
        margin=dict(l=0, r=0, b=0, t=30),
        title=dict(text=f"{run.name}   {a.node}", x=0.5, font=dict(size=13)),
        scene=dict(
            xaxis=dict(title=dict(text="Temperature (°C)")),
            yaxis=dict(title=dict(text="Precipitation (m)")),
            zaxis=dict(title=dict(text="Δ ln(GDP per capita)" if levels else "Δ ln(Growth)"),
                       range=[zlo - pad, zhi + pad]),
            camera=CAMERA,
        ),
        updatemenus=[dict(
            type="buttons", showactive=False, x=-0.01, xanchor="right", y=0.05,
            pad=dict(r=10, t=60),
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=150, redraw=True),
                                      transition=dict(duration=0),
                                      fromcurrent=True, mode="immediate")]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode="immediate",
                                        transition=dict(duration=0))]),
            ])],
        sliders=[dict(active=0, currentvalue=dict(prefix="Year: "),
                      pad=dict(t=50), steps=steps)],
    )

    out = Path(a.outfile) if a.outfile else (
        ROOT / "results/images" / f"dynamic_animation_{run.name}_{a.node}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    # Self-contained by default: the file embeds plotly.js so it opens with no
    # internet connection and survives being emailed or dropped in a shared folder.
    fig.write_html(out, include_plotlyjs=("cdn" if a.cdn else True), auto_play=False)
    print(f"\n  -> {out}  ({out.stat().st_size/1e6:.1f} MB"
          f"{', CDN' if a.cdn else ', self-contained'})")


if __name__ == "__main__":
    main()
