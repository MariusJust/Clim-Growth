"""Initialisation stability for the 2026-09-26 static and dynamic runs.

Companion to notes/2026-09-27/2026-09-27-init-stability-analysis.md, which
already reports the BIC side from initialisations.csv. This script adds the part
that needs the per-initialisation weight snapshots:

  1. full 12-cell BIC table (fills the three static cells the note leaves blank)
  2. best-of-10 block analysis, removing the order-statistic advantage that
     best-of-40 has over the older 10-init runs
  3. nesting check on the nested pairs available here:
     (2,) in (4,), (2,2) in (4,4), (2,2,2) in (4,2,2)
  4. surface stability across initialisations: optimum temperature, amplitude,
     and for the dynamic run the time-varying share and corr(1961, 2023)
  5. the three figures the paper1-analysis skill mandates, for both runs

Everything except step 5 is TensorFlow-free: the climate net is a small swish
MLP evaluated with numpy, reading the Keras weights through scripts/hdf5_min.py.

    python scripts/2026-09-27-init-stability-analysis.py
    python scripts/2026-09-27-init-stability-analysis.py --bic-tol 5 --no-figures

Writes results/tables/2026-09-27-init-stability-*.csv and prints every table.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from scripts.archive.hdf5_min import read_weights  # noqa: E402

STATIC = "runs/estimation/2026-09-26_18-22-33_global_IC_country_trends"
DYNAMIC = "runs/estimation/2026-09-26_18-23-41_global_dynamic"
OUT_TAB = ROOT / "results/tables"


# ----------------------------------------------------------------------------
# model evaluation, no TensorFlow
# ----------------------------------------------------------------------------

def _within_projector():
    """Import WithinProjector by path: the package __init__ chain pulls in TF."""
    spec = importlib.util.spec_from_file_location(
        "within_projection",
        ROOT / "src/models/helper_functions/global_model/within_projection.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod.WithinProjector


def swish(z):
    # exp overflows for very negative z; clip keeps the identity branch exact.
    return z / (1.0 + np.exp(-np.clip(z, -60.0, 60.0)))


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


class Run:
    """One run directory: config, panel, within projector and evaluation grids."""

    def __init__(self, rel):
        self.dir = ROOT / rel
        cfg_file = self.dir / ".hydra/config.yaml"
        if not cfg_file.is_file():
            cfg_file = self.dir / "config/config.yaml"
        cfg = yaml.safe_load(cfg_file.read_text(encoding="utf-8"))["instance"]
        m = cfg["model"]
        self.dynamic = bool(m.get("dynamic_model", False))
        self.ct = bool(m.get("country_trends", False))
        self.qt = bool(m.get("quadratic_trends", self.ct))
        self.levels = str(m.get("target_mode", "growth")).lower() == "levels"

        d = pd.read_excel(ROOT / "data/MainData.xlsx")
        if m.get("data_end"):
            d = d[d.Year <= m["data_end"]]
        Tp = d.pivot(index="Year", columns="CountryCode", values="TempPopWeight")
        Pp = d.pivot(index="Year", columns="CountryCode",
                     values="PrecipPopWeight").reindex_like(Tp)
        Yp = d.pivot(index="Year", columns="CountryCode",
                     values="GDPCap" if self.levels else "GrowthWDI").reindex_like(Tp)
        self.Tv, self.Pv = Tp.to_numpy(float), Pp.to_numpy(float)
        self.Yv = np.log(Yp.to_numpy(float)) if self.levels else Yp.to_numpy(float)
        self.years = np.asarray(Tp.index, dtype=int)
        self.nT, self.nN = self.Tv.shape

        self.mT, self.sT = float(np.nanmean(self.Tv)), float(np.nanstd(self.Tv))
        self.mP, self.sP = float(np.nanmean(self.Pv)), float(np.nanstd(self.Pv))
        self.mt = (self.nT + 1) / 2
        self.st = np.sqrt((self.nT ** 2 - 1) / 12)

        mask = np.isnan(self.Tv)
        self.n = int((~mask).sum())
        WP = _within_projector()
        self.pr = WP(mask, country_trends=self.ct, quadratic_trends=self.qt,
                     include_time=not self.dynamic)

        # 5th to 95th percentile of observed temperature: only 3 of 10,324
        # country-years exceed 30 C, so the tails are functional form, not data.
        self.t_lo, self.t_hi = np.nanpercentile(self.Tv, [5, 95])
        self.p_med = float(np.nanmedian(self.Pv))
        self.k = 3 if self.dynamic else 2

    # -- full-panel design, for MSE ------------------------------------------
    def panel_X(self):
        cols = [np.nan_to_num((self.Tv - self.mT) / self.sT),
                np.nan_to_num((self.Pv - self.mP) / self.sP)]
        if self.dynamic:
            cols.append(((np.arange(1, self.nT + 1)[:, None] - self.mt) / self.st)
                        * np.ones((1, self.nN)))
        return np.stack(cols, axis=-1)

    def mse(self, w):
        """Within-MSE after recovering the nuisance coefficients by exact OLS."""
        f_full = forward(w, self.panel_X())
        yo = self.Yv[self.pr.t_arr, self.pr.n_arr]
        fo = f_full[self.pr.t_arr, self.pr.n_arr]
        ok = np.isfinite(yo) & np.isfinite(fo)
        g = self.pr.recover_gamma(np.where(ok, yo - fo, 0.0))
        resid = np.where(ok, yo, 0.0) - (np.where(ok, fo, 0.0) + self.pr.W @ g)
        return float(np.sum(resid ** 2) / self.n)

    # -- response surface on the reporting grid -------------------------------
    def surface(self, w, n_temp=120):
        """Centred response in pp. Shape (n_temp, nT) if dynamic else (n_temp,)."""
        Tg = np.linspace(self.t_lo, self.t_hi, n_temp)
        if self.dynamic:
            TT, tt = np.meshgrid(Tg, np.arange(1, self.nT + 1), indexing="ij")
            X = np.stack([(TT - self.mT) / self.sT,
                          np.full_like(TT, (self.p_med - self.mP) / self.sP),
                          (tt - self.mt) / self.st], axis=-1)
            G = forward(w, X)
            return Tg, (G - G.mean(axis=0, keepdims=True)) * 100.0
        X = np.stack([(Tg - self.mT) / self.sT,
                      np.full_like(Tg, (self.p_med - self.mP) / self.sP)], axis=-1)
        G = forward(w, X)
        return Tg, (G - G.mean()) * 100.0

    def n_params(self, node):
        m, p = 0, self.k
        for h in node:
            m += p * h + h
            p = h
        return m + p


# ----------------------------------------------------------------------------
# tables
# ----------------------------------------------------------------------------

def load_inits(run: Run):
    out = {}
    for nd in sorted((run.dir / "diagnostics").iterdir()):
        f = nd / "initialisations.csv"
        if f.is_file():
            out[nd.name] = pd.read_csv(f).set_index("init").sort_index()
    return out


def bic_table(frames, label):
    rows = []
    for node, df in frames.items():
        b = pd.to_numeric(df["BIC"], errors="coerce").dropna().to_numpy()
        lr = pd.to_numeric(df.get("lr_final"), errors="coerce")
        ep = pd.to_numeric(df.get("epochs"), errors="coerce")
        r2 = pd.to_numeric(df.get("R2"), errors="coerce")
        rows.append(dict(arch=node, n=len(b), BIC_best=b.min(), BIC_worst=b.max(),
                         spread=b.max() - b.min(), sd=b.std(ddof=1),
                         lr_at_min=int((lr <= 1.1e-7).sum()),
                         ep_min=int(ep.min()), ep_max=int(ep.max()),
                         R2_min=r2.min(), R2_max=r2.max()))
    t = pd.DataFrame(rows).sort_values("BIC_best").reset_index(drop=True)
    t["dBIC"] = t.BIC_best - t.BIC_best.min()
    print(f"\n=== {label}: BIC over initialisations ===")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    bad = t[t.lr_at_min < t.n]
    print("  annealing: ALL initialisations reached lr_min" if bad.empty
          else f"  WARNING: {len(bad)} cells did not reach lr_min:\n{bad[['arch','n','lr_at_min']]}")
    return t


def block_table(frames, label, block=10):
    rows = []
    for node, df in frames.items():
        b = pd.to_numeric(df["BIC"], errors="coerce").to_numpy()
        mins = [np.nanmin(b[i:i + block]) for i in range(0, len(b), block)]
        r = {"arch": node}
        r.update({f"blk{i}": v for i, v in enumerate(mins)})
        r["blk_spread"] = max(mins) - min(mins)
        rows.append(r)
    t = pd.DataFrame(rows).sort_values("blk0").reset_index(drop=True)
    print(f"\n=== {label}: min BIC per independent block of {block} ===")
    print("  (removes the order-statistic advantage of best-of-40 over best-of-10)")
    print(t.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return t


def nesting_table(run: Run, label):
    """MSE must be monotone in width at a true optimum: a wider net can copy a
    narrower one and zero the remainder. Any violation is non-convergence."""
    rows = []
    for f in sorted((run.dir / "parameters").glob("*.weights.h5")):
        node = ast.literal_eval(f.stem)
        rows.append(dict(node=node, params=run.n_params(node),
                         MSE=run.mse(read_weights(str(f)))))
    df = pd.DataFrame(rows)
    nests = lambda a, b: len(a) == len(b) and all(x <= y for x, y in zip(a, b)) and a != b
    viol = [(r.node, r.MSE, s.node, s.MSE, (s.MSE / r.MSE - 1) * 100)
            for _, r in df.iterrows() for _, s in df.iterrows()
            if nests(r.node, s.node) and s.MSE > r.MSE]
    pairs = sum(1 for _, r in df.iterrows() for _, s in df.iterrows() if nests(r.node, s.node))
    print(f"\n=== {label}: nesting check ===")
    print(df.sort_values("MSE").to_string(index=False, float_format=lambda x: f"{x:.8g}"))
    print(f"  nested pairs available: {pairs};  violations: {len(viol)}")
    for a, ma, b, mb, pc in sorted(viol, key=lambda x: -x[4]):
        print(f"    {str(a):>12} {ma:.8g}  <  {str(b):>12} {mb:.8g}  ({pc:+.2f}% worse)")
    return df, viol


def surface_stability(run: Run, frames, label, bic_tol):
    """Per-initialisation surface geometry, for fits that are equally good on BIC."""
    recs = []
    for node, df in frames.items():
        d = run.dir / "diagnostics" / node
        best = pd.to_numeric(df["BIC"], errors="coerce").min()
        for j in df.index:
            f = d / f"init_{j}.weights.h5"
            if not f.is_file():
                continue
            bic = float(df.loc[j, "BIC"])
            Tg, G = run.surface(read_weights(str(f)))
            rec = dict(arch=node, init=int(j), BIC=bic, dBIC=bic - best)
            if run.dynamic:
                gbar = G.mean(axis=1, keepdims=True)
                r = G - gbar
                rec.update(opt_T=float(Tg[np.argmax(gbar[:, 0])]),
                           amp_pp=float(gbar.max() - gbar.min()),
                           tv_pct=float((r ** 2).sum() / (G ** 2).sum() * 100),
                           corr_first_last=float(np.corrcoef(G[:, 0], G[:, -1])[0, 1]))
            else:
                rec.update(opt_T=float(Tg[np.argmax(G)]),
                           amp_pp=float(G.max() - G.min()),
                           tv_pct=np.nan, corr_first_last=np.nan)
            recs.append(rec)
    full = pd.DataFrame(recs)

    cols = ["opt_T", "amp_pp"] + (["tv_pct", "corr_first_last"] if run.dynamic else [])
    summ = []
    for node, g in full.groupby("arch"):
        keep = g[g.dBIC <= bic_tol]
        row = dict(arch=node, n_inits=len(g), n_within_tol=len(keep))
        for c in cols:
            v = keep[c].to_numpy(float)
            row[f"{c}_min"], row[f"{c}_max"] = v.min(), v.max()
            row[f"{c}_spread"] = v.max() - v.min()
        summ.append(row)
    s = pd.DataFrame(summ).sort_values("arch").reset_index(drop=True)
    print(f"\n=== {label}: surface geometry across inits within {bic_tol} BIC of best ===")
    print(s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return full, s


def mandated_figures(run_rel, skill_dir):
    if skill_dir is None:
        print("\n  skill scripts/ not found, skipping the three mandated figures."
              "\n  Pass --skill-dir <paper1-analysis/scripts> to generate them.")
        return
    sys.path.insert(0, str(skill_dir))
    try:
        from make_figures import make_all
    except Exception as exc:                                        # noqa: BLE001
        print(f"\n  could not import make_figures ({type(exc).__name__}: {exc}); skipping")
        return
    print(f"\n=== mandated figures for {run_rel} ===")
    try:
        make_all(run_rel, outdir="results/images", top=20)
    except Exception as exc:                                        # noqa: BLE001
        print(f"  make_all failed: {type(exc).__name__}: {exc}")


def find_skill_dir(explicit):
    if explicit:
        p = Path(explicit)
        return p if p.is_dir() else None
    for c in [ROOT / ".claude/skills/paper1-analysis/scripts",
              ROOT.parent / ".claude/skills/paper1-analysis/scripts"]:
        if c.is_dir():
            return c
    hits = list(Path.home().glob(
        "**/skills/paper1-analysis/scripts"))
    return hits[0] if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bic-tol", type=float, default=2.0,
                    help="inits within this many BIC of the cell best count as equally good")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--skill-dir", default=None)
    a = ap.parse_args()

    OUT_TAB.mkdir(parents=True, exist_ok=True)
    skill = None if a.no_figures else find_skill_dir(a.skill_dir)

    for rel, label in ((STATIC, "STATIC 2026-09-26_18-22-33"),
                       (DYNAMIC, "DYNAMIC 2026-09-26_18-23-41")):
        print("\n" + "=" * 78)
        print(label)
        print("=" * 78)
        run = Run(rel)
        print(f"  dynamic={run.dynamic}  country_trends={run.ct}  levels={run.levels}"
              f"  n={run.n}  k={run.k}  grid {run.t_lo:.1f}-{run.t_hi:.1f} C")
        frames = load_inits(run)

        tag = "static" if not run.dynamic else "dynamic"
        bic_table(frames, label).to_csv(OUT_TAB / f"2026-09-27-init-stability-bic-{tag}.csv", index=False)
        block_table(frames, label).to_csv(OUT_TAB / f"2026-09-27-init-stability-blocks-{tag}.csv", index=False)
        nesting_table(run, label)
        full, summ = surface_stability(run, frames, label, a.bic_tol)
        full.to_csv(OUT_TAB / f"2026-09-27-init-stability-surfaces-{tag}.csv", index=False)
        summ.to_csv(OUT_TAB / f"2026-09-27-init-stability-surface-summary-{tag}.csv", index=False)

        if not a.no_figures:
            mandated_figures(rel, skill)

    print(f"\ntables -> {OUT_TAB}")


if __name__ == "__main__":
    main()
