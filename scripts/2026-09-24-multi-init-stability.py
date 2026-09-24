"""How much does the estimated response surface move across initialisations?

Motivation: two levels runs differing only in the output-layer initialiser sit
1.03 BIC apart yet place the growth-maximizing temperature 18 C apart (11.67 C
under he_normal, 30.00 C under zeros, the latter pinned at the grid edge). That
is either two draws from a wide distribution over local optima, or two isolated
optima. This script answers which.

    python scripts/2026-09-24-multi-init-stability.py --run <run_dir> [--node "(2,)"]
    python scripts/2026-09-24-multi-init-stability.py --compare <dir1> <dir2> ...

Requires the run to have been estimated with ``use_diagnostics: true``, which
writes ``<run>/diagnostics/<node>/init_<j>.weights.h5`` for every initialisation
together with ``initialisations.csv`` holding their BIC, AIC and R2. The run's
own config supplies target_mode and dynamic_model, so growth, levels and dynamic
runs are all handled by the same call.

For every initialisation it reports the optimum temperature and the response
range, both over the full 0-30 C grid and restricted to the 5th-95th percentile
of observed temperature, since an argmax pinned at a grid edge is an artefact of
the unsupported tail rather than a finding. The headline number is the spread of
the optimum among initialisations whose BIC is within ``--tol`` of that cell's
best: those are the fits a reader would regard as equally good.
"""
from __future__ import annotations

import argparse
import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from hdf5_min import read_weights  # noqa: E402

OUT_TAB = ROOT / "paper/Tables"


def _out_fig() -> Path:
    """Preferred figure directory, falling back to one that already exists.

    Creating a new directory does not persist on the OneDrive-backed mount used
    by the sandbox: mkdir appears to succeed in-process and the directory is
    gone afterwards, so savefig fails with FileNotFoundError. Probe for real
    persistence rather than trusting mkdir, and fall back to an existing folder.
    """
    pref = ROOT / "paper/Figures/Diagnostics"
    try:
        pref.mkdir(parents=True, exist_ok=True)
        probe = pref / ".writable"
        probe.write_text("")
        if pref.is_dir():
            return pref
    except OSError:
        pass
    fallback = ROOT / "paper/Figures/Misc"
    print(f"note: cannot create {pref.relative_to(ROOT)}; writing to "
          f"{fallback.relative_to(ROOT)} instead")
    return fallback


def swish(z):
    return z / (1.0 + np.exp(-z))


def forward(w, X):
    idx = sorted({int(k.split("/")[1].split("_")[1]) if "_" in k.split("/")[1] else 0
                  for k in w if k.startswith("layers")})
    h = X
    for i in idx:
        b = f"layers/dense{'' if i == 0 else f'_{i}'}/vars"
        h = h @ np.asarray(w[f"{b}/0"], float)
        if f"{b}/1" in w:
            h = swish(h + np.asarray(w[f"{b}/1"], float))
    return h[..., 0]


def load_panel():
    d = pd.read_excel(ROOT / "data/MainData.xlsx")
    d = d[d.Year <= 2024]
    Tp = d.pivot(index="Year", columns="CountryCode", values="TempPopWeight")
    Pv = (d.pivot(index="Year", columns="CountryCode", values="PrecipPopWeight")
            .reindex_like(Tp).to_numpy(float))
    Tv = Tp.to_numpy(float)
    return Tv, Pv, len(Tp.index)


def cell(run_dir: Path, node_str: str | None, tol: float):
    cfg = yaml.safe_load((run_dir / ".hydra/config.yaml").read_text(encoding="utf-8"))["instance"]
    m = cfg.get("model", {})
    dyn = bool(m.get("dynamic_model", False))
    levels = str(m.get("target_mode", "growth")).lower() == "levels"
    init = m.get("output_initializer") or "zeros (code default)"

    ddir = run_dir / "diagnostics"
    if not ddir.is_dir():
        raise SystemExit(f"{run_dir.name}: no diagnostics/ directory. The run needs "
                         "use_diagnostics: true; per-init weights were not saved.")
    nodes = sorted(p for p in ddir.iterdir() if p.is_dir())
    if node_str:
        nodes = [p for p in nodes if p.name == node_str]
    if not nodes:
        raise SystemExit(f"{run_dir.name}: no matching node under diagnostics/")
    ndir = nodes[0]
    node = tuple(ast.literal_eval(ndir.name))

    csv = ndir / "initialisations.csv"
    meta = pd.read_csv(csv) if csv.exists() else None

    Tv, Pv, nT = load_panel()
    mT, sT = float(np.nanmean(Tv)), float(np.nanstd(Tv))
    mP, sP = float(np.nanmean(Pv)), float(np.nanstd(Pv))
    Pmed = float(np.nanmedian(Pv))
    lo, hi = np.nanpercentile(Tv, [5, 95])
    Tfull = np.linspace(0, 30, 1201)
    sup = (Tfull >= lo) & (Tfull <= hi)
    # dynamic surfaces are evaluated at the mid-sample date; the time dimension
    # is a separate question handled by 2026-09-08-dynamic-time-surface.py
    tz_mid = 0.0

    rows, curves = [], []
    for f in sorted(ndir.glob("init_*.weights.h5"),
                    key=lambda p: int(re.search(r"init_(\d+)", p.name).group(1))):
        j = int(re.search(r"init_(\d+)", f.name).group(1))
        w = read_weights(str(f))
        cols = [(Tfull - mT) / sT, np.full_like(Tfull, (Pmed - mP) / sP)]
        if dyn:
            cols.append(np.full_like(Tfull, tz_mid))
        y = forward(w, np.stack(cols, axis=-1))
        y = y - y[int(np.argmin(np.abs(Tfull - mT)))]        # centre at sample mean
        b = np.nan
        if meta is not None and "init" in meta.columns and (meta.init == j).any():
            b = float(meta.loc[meta.init == j, "BIC"].iloc[0])
        rows.append(dict(init=j, BIC=b,
                         opt_full=Tfull[int(np.argmax(y))],
                         opt_sup=Tfull[sup][int(np.argmax(y[sup]))],
                         range_full=(y.max() - y.min()) * 100,
                         range_sup=(y[sup].max() - y[sup].min()) * 100,
                         edge=bool(np.argmax(y) in (0, len(Tfull) - 1))))
        curves.append(y * 100)

    df = pd.DataFrame(rows)
    label = (f"{'levels' if levels else 'growth'}"
             f"{' dynamic' if dyn else ''} {ndir.name}, init={init}")
    print(f"\n=== {label} ===")
    print(f"    {run_dir.name}   {len(df)} initialisations")
    if df.BIC.notna().any():
        best = df.BIC.min()
        keep = df[df.BIC <= best + tol]
        print(f"    BIC best {best:.2f}   within {tol:g}: {len(keep)}/{len(df)} inits")
    else:
        keep = df
        print("    no initialisations.csv; reporting over all inits")
    for nm, col, unit in (("optimum, supported range", "opt_sup", "C"),
                          ("optimum, full 0-30 grid", "opt_full", "C"),
                          ("response range, supported", "range_sup", "pp"),
                          ("response range, full grid", "range_full", "pp")):
        v = keep[col]
        print(f"    {nm:28} min {v.min():7.2f}{unit}  max {v.max():7.2f}{unit}  "
              f"median {v.median():7.2f}{unit}  sd {v.std():6.2f}")
    print(f"    argmax at a grid edge in {int(keep.edge.sum())}/{len(keep)} inits")
    return df, np.array(curves), label, Tfull, (lo, hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run")
    ap.add_argument("--compare", nargs="+")
    ap.add_argument("--node", default=None)
    ap.add_argument("--tol", type=float, default=10.0,
                    help="BIC tolerance defining 'equally good' fits (default 10)")
    a = ap.parse_args()
    runs = a.compare or ([a.run] if a.run else [])
    if not runs:
        raise SystemExit("give --run <dir> or --compare <dir> <dir> ...")

    OUT_FIG = _out_fig()
    OUT_TAB.mkdir(parents=True, exist_ok=True)
    allrows, fig, axes = [], None, None
    n = len(runs)
    fig, axes = plt.subplots(1, n, figsize=(5.0 * n, 4.0), squeeze=False)

    for k, r in enumerate(runs):
        df, curves, label, Tg, (lo, hi) = cell(Path(r) if Path(r).is_absolute()
                                               else ROOT / r, a.node, a.tol)
        df.insert(0, "cell", label)
        allrows.append(df)
        ax = axes[0][k]
        for c in curves:
            ax.plot(Tg, c, color="0.6", lw=0.7, alpha=0.6)
        if df.BIC.notna().any():
            ax.plot(Tg, curves[int(df.BIC.idxmin())], color="#2f6f9f", lw=2.0,
                    label="BIC-best init")
            ax.legend(fontsize=8, frameon=False)
        ax.axvspan(0, lo, color="0.92", zorder=0)
        ax.axvspan(hi, 30, color="0.92", zorder=0)
        ax.axhline(0, color="k", lw=0.5)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("temperature ($\\degree$C)")
        ax.set_ylabel("response, centred at sample mean (pp)")

    fig.tight_layout()
    dest = OUT_FIG / "2026-09-24-multi-init-stability.pdf"
    fig.savefig(dest, bbox_inches="tight")
    fig.savefig(dest.with_suffix(".png"), dpi=170, bbox_inches="tight")
    out = pd.concat(allrows, ignore_index=True)
    out.to_csv(OUT_TAB / "2026-09-24-multi-init-stability.csv", index=False)
    print(f"\nfigure -> {dest}")
    print(f"table  -> {OUT_TAB/'2026-09-24-multi-init-stability.csv'}")
    print("\nShaded bands mark temperature outside the 5th-95th percentile of the "
          "sample, where the surface is functional form rather than evidence.")


if __name__ == "__main__":
    main()
