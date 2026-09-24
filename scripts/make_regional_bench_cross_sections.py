"""Appendix M figure: per-region temperature cross-sections of the REGIONAL
neural network vs. the Burke (2015) quadratic benchmark.

The neural-network curves are cut directly from the regional model of
Figure~\\ref{fig:regional_model_1} -- i.e. each panel's blue line is a slice of
the corresponding Figure 6 surface -- so Appendix M and Figure 6 show the same
network. This is the whole point of the script: use the AUTHORITATIVE
region->surface mapping (``model.model_visual[region]``, keyed by region name)
rather than guessing which saved layer belongs to which region. That mapping
only exists once the Keras model is instantiated, so this MUST run on a
TF-enabled machine (your server / devcontainer), NOT in a TF-free sandbox.

What it does, per region:
  * NN: evaluate ``model.model_visual[region]`` on the shared (T,P) grid, slice
    at that region's MEDIAN observed (population-weighted) precipitation, and
    centre at that region's MEAN temperature.
  * Burke: re-estimate the quadratic on that region's SUBSAMPLE with the same
    fixed effects and country trends (``fit_burke``), evaluate its climate-only
    surface along temperature at the same median precipitation, and centre at
    the same mean temperature.

Architecture is forced to (4,) by default (``--node "(4,)"``), matching the
"four hidden nodes" regional model in Section 5. Change --node if you switch
the reported architecture.

Run from the repo root (devcontainer sets PYTHONPATH=src)::

    PYTHONPATH=src python scripts/make_regional_bench_cross_sections.py \\
        runs/estimation/2026-06-16_19-10-06_regional_IC_country_trends \\
        --node "(4,)" \\
        --out paper/Figures/Results/regional_bench_cross_sections_burke.pdf

Requires: tensorflow/keras, h5py, statsmodels, pandas, numpy, matplotlib.
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Panel order, left -> right, matching the current Appendix M figure.
PANEL_ORDER = ["Europe", "Africa", "Asia", "Oceania", "Americas"]

# UN M49 region codes as used in MainData.RegionCode.
REGION_CODE = {"Asia": 142, "Europe": 150, "Africa": 2, "Americas": 19,
               "Oceania": 9}


def _load_cfg(run_dir: Path) -> dict:
    with open(run_dir / ".hydra" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["instance"]


def _benchmark_frame(sub: pd.DataFrame) -> pd.DataFrame:
    """Rename a region subsample to the columns fit_burke expects.

    Temperature in degC, precipitation in METRES (PrecipPopWeight is mm),
    growth = GrowthWDI. The pre-computed per-country trend columns
    (X_yi_*/X_y2_*) are carried through and auto-detected by bench_models.
    """
    d = sub.rename(columns={"TempPopWeight": "temp"}).copy()
    d["precip"] = sub["PrecipPopWeight"].values / 1000.0
    d["growth"] = sub["GrowthWDI"].values
    return d


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir",
                    help="regional run dir, e.g. "
                         "runs/estimation/2026-06-16_19-10-06_regional_IC_country_trends")
    ap.add_argument("--node", default="(4,)",
                    help='architecture string; default "(4,)"')
    ap.add_argument("--out",
                    default="paper/Figures/Results/regional_bench_cross_sections_burke.pdf",
                    help="output PDF path")
    ap.add_argument("--legend-arch", default=None,
                    help="legend label for the NN line; default derived from --node")
    ap.add_argument("--zrange", type=float, nargs=2, default=(-0.4, 0.3),
                    metavar=("LO", "HI"),
                    help="fixed vertical (Delta ln GDP) axis range for every "
                         "panel; default -0.4 0.3, matching the z-axis of the "
                         "Figure 6 surfaces. Override if the surface range "
                         "changes.")
    args = ap.parse_args()

    repo = Path(args.run_dir).resolve().parents[2]
    src = repo / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    # Imports that need TF / statsmodels live here so --help works anywhere.
    from models import MultivariateModelRegional as Model
    from models.helper_functions.regional_model.prepare_data import Prepare
    from utils import create_pred_input
    from utils.miscelaneous import Find_data_file

    # bench_models is loaded directly by file path, NOT via
    # ``from simulations...``: importing the ``simulations`` package runs its
    # __init__, which eagerly imports monte_carlo -> ``utils.config``
    # (currently missing), and that unrelated breakage would abort us.
    # bench_models.py itself only needs numpy/pandas/statsmodels, so a
    # standalone file load is safe and side-steps the package entirely.
    import importlib.util
    _bm_path = src / "simulations" / "simulation_functions" / "bench_models.py"
    _spec = importlib.util.spec_from_file_location("bench_models", _bm_path)
    _bm = importlib.util.module_from_spec(_spec)
    # Register in sys.modules BEFORE exec: @dataclass resolves its class's
    # module via sys.modules[cls.__module__]; if the module isn't registered
    # that lookup returns None and dataclass creation raises AttributeError.
    sys.modules[_spec.name] = _bm
    _spec.loader.exec_module(_bm)
    fit_burke = _bm.fit_burke
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    run_dir = Path(args.run_dir)
    cfg = _load_cfg(run_dir)
    node = tuple(int(x) for x in ast.literal_eval(args.node))
    node_label = args.legend_arch or f"NN {args.node}"
    print(f"run_dir: {run_dir}\narchitecture: {node}")

    # ---- data + grid (mirrors make_regional_figures.py) --------------------
    if str(cfg.get("data_source", "wb")).lower() == "ee":
        data = pd.read_csv(Find_data_file("ee_data.csv"), sep=";")
    else:
        data = pd.read_excel(Find_data_file("MainData.xlsx"))
    if cfg.get("data_end") is not None and "Year" in data:
        data = data[data["Year"] <= cfg["data_end"]]

    growth, precip, temp = Prepare(data, data_source=cfg["data_source"],
                                   formulation=cfg["formulation"])
    x_train = {0: temp, 1: precip}

    tcol = "TempPopWeight" if "TempPopWeight" in data else "temperature (celsius)"
    pcol = "PrecipPopWeight" if "PrecipPopWeight" in data else "precipitation (mm)"

    # PER-REGION standardisation. Prepare() z-scores temperature and precipitation
    # inside its per-region loop, so region r's sub-network is trained in region
    # r's own units and Visual_model applies no standardisation of its own.
    # Feeding a single globally standardised grid to every head (what this script
    # did before 2026-09-04) evaluates each region at the wrong point of its
    # domain -- for Europe, 20 C is z=+2.68 in its own units but z=-0.04 globally.
    # Verified: per-region moments reproduce the (2,) run's recorded BIC to 0.025
    # out of -35,474.94; global moments are off by 105.6.
    if str(Path(__file__).resolve().parent) not in sys.path:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
    from regional_surface import region_moments          # noqa: E402
    reg_mom = region_moments(data)
    pred_inputs = {}
    for _r, (_mT, _sT, _mP, _sP) in reg_mom.items():
        pred_inputs[_r], T_grid, P_grid = create_pred_input(
            mc=False, mean_T=_mT, std_T=_sT, mean_P=_mP, std_P=_sP,
            precip_capped=True)
    temp_axis = T_grid[0, :]       # degC (columns) -- raw units, shared across regions
    prec_axis = P_grid[:, 0]       # mm   (rows)

    # ---- build regional model, load the chosen weights ---------------------
    factory = Model(node, cfg, x_train, growth)
    factory.Depth = len(node)
    model = factory.get_model()
    weights = run_dir / "parameters" / f"{node}.weights.h5"
    model.load_params(str(weights))
    print(f"loaded {weights}")

    # ---- figure ------------------------------------------------------------
    fig, axes = plt.subplots(1, len(PANEL_ORDER), figsize=(15, 3.0))
    for ax, region in zip(axes, PANEL_ORDER):
        if region not in model.model_visual:
            raise SystemExit(f"region '{region}' not in model.model_visual; "
                             f"available: {list(model.model_visual)}")

        # NN surface for this region, evaluated in THIS region's z-units.
        flat = model.model_visual[region].predict([pred_inputs[region]],
                                                  verbose=0).reshape(-1)
        Z = flat.reshape(T_grid.shape)

        # Region subsample -> median precip (mm) and mean temp (degC).
        sub = data[data["RegionCode"] == REGION_CODE[region]]
        med_P_mm = float(np.nanmedian(sub[pcol]))
        mean_T_r = float(np.nanmean(sub[tcol]))
        prec_idx = int(np.argmin(np.abs(prec_axis - med_P_mm)))
        temp_idx = int(np.argmin(np.abs(temp_axis - mean_T_r)))
        med_P_m = med_P_mm / 1000.0

        # NN cross-section, centred at the region's mean temperature.
        nn_curve = Z[prec_idx, :].copy()
        nn_curve -= nn_curve[temp_idx]

        # Burke on the region subsample, same FE + trends, same centring.
        burke = fit_burke(_benchmark_frame(sub),
                          country_trends=cfg.get("country_trends", True),
                          quadratic_trends=cfg.get("quadratic_trends", True))
        burke_curve = np.asarray(burke.predict_surface(temp_axis, med_P_m), float)
        burke_curve -= float(burke.predict_surface(temp_axis[temp_idx], med_P_m))

        ax.plot(temp_axis, nn_curve, color="steelblue", lw=2, label=node_label)
        ax.plot(temp_axis, burke_curve, color="firebrick", lw=2, ls="--",
                label="Burke")
        ax.axhline(0, color="k", lw=0.8, ls=":")
        ax.axvline(mean_T_r, color="grey", lw=0.8, ls=":")
        ax.set_title(f"{region} (P={med_P_m:.2f} m)", fontsize=10)
        ax.set_xlabel("Temperature (°C)", fontsize=9)
        ax.set_xlim(0, 30)
        # Same fixed vertical scale on every panel as the Figure 6 surface
        # z-axis, so Appendix M and Figure 6 are read on identical scales.
        # NB: like the Figure 6 surfaces (which set the same plotly zaxis
        # range), a curve running past this range is clipped at the frame;
        # widen with --zrange if you would rather show the full excursion.
        ax.set_ylim(*args.zrange)
        if region == PANEL_ORDER[0]:
            ax.set_ylabel(r"$\Delta \ln$(GDP), centered", fontsize=9)
            ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
