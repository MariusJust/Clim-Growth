"""Local pre-flight for the dynamic model, BEFORE submitting on the server.

Needs TensorFlow, so run it on your machine or a login node with the env active:

    PYTHONPATH=src python scripts/test_dynamic_preflight.py
    PYTHONPATH=src python scripts/test_dynamic_preflight.py --node "(4,4,4)" --patience 30

It answers the two questions the sandbox could not, plus runs the actual
experiment in miniature so you know whether 19 hours is worth spending:

  1 CONFIG      does the run resolve to the settings you think it does?
  2 TIME SCALE  read time_mu / time_sigma straight off the Vectorize layer the
                model actually built. This is the definitive check that the
                standardisation is live - not inferred from weight magnitudes.
  3 INIT        read the output kernel BEFORE training. 'zeros' must give
                exactly 0; 'he_normal' must give non-zero. With a zero kernel
                and use_bias=False the hidden layers get dL/dh = 0 on step one.
  4 SHORT FIT   fit the same architecture twice, once per initialiser, with a
                small patience. Reports epochs, loss, BIC, and - the thing that
                matters - the temperature range of the fitted surface and
                whether it has an interior optimum.

This is a SHORT fit (small patience), so the numbers are not production values.
It tells you whether the pipeline runs and whether the initialiser moves the
result, not what the final answer is.

Writes nothing to runs/; no weights are saved.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

TREF = [1961, 1990, 2020]


def build_and_fit(inst, node, patience, min_delta, verbose=0):
    """Mirror run_experiment_ic's _build_model_instance + one initialisation."""
    import random
    import tensorflow as tf
    from models import MultivariateModelGlobal as Model
    from models.helper_functions.global_model import load_data

    seed = int(getattr(inst, "seed_value", 0))
    tf.random.set_seed(seed); np.random.default_rng(seed); random.seed(seed)

    growth, precip, temp = load_data(
        "IC", inst.data_source, end_year=inst.data_end,
        target_mode=getattr(inst, "target_mode", "growth"))

    factory = Model(node=None, cfg=inst, x_train=None, y_train=None,
                    x_train_val=None, y_train_val=None, x_val=None, y_val=None)
    factory.country_map = None
    factory.x_train = {0: temp, 1: precip}
    factory.y_train = growth
    factory.node = node

    mf = Model(node=node, cfg=inst,
               x_train=factory.x_train, y_train=factory.y_train,
               x_train_val=factory.x_train_val, y_train_val=factory.y_train_val,
               x_val=factory.x_val, y_val=factory.y_val)
    mf.country_map = None
    m = mf.get_model()

    # --- inspect BEFORE training -----------------------------------------
    w_out0 = np.asarray(m.output_layer.weights[0].numpy())
    tmu = tsig = None
    for lyr in m.model.layers:
        if getattr(lyr, "variable", None) == "time":
            tmu, tsig = getattr(lyr, "time_mu", None), getattr(lyr, "time_sigma", None)
            break

    t0 = time.time()
    m.fit(lr=inst.lr, min_delta=min_delta, patience=patience, verbose=verbose)
    m.in_sample_predictions()
    secs = time.time() - t0
    epochs = len(m.model.history.history.get("loss", []))
    return m, dict(w_out0=w_out0, time_mu=tmu, time_sigma=tsig,
                   secs=secs, epochs=epochs,
                   loss=float(m.model.history.history["loss"][-1]),
                   BIC=float(m.BIC), AIC=float(m.AIC),
                   m_eff=int(getattr(m, "m_effective", -1) or -1))


def surface_stats(m, inst, tmu, tsig):
    """Temperature cross-section at mean precipitation, per year."""
    layers = []
    for nm in ("hidden_1", "hidden_2", "hidden_3"):
        l = getattr(m, nm, None)
        if l is not None:
            W = np.asarray(l.weights[0].numpy())
            b = np.asarray(l.weights[1].numpy()) if len(l.weights) > 1 else None
            layers.append((W, b))
    Wo = np.asarray(m.output_layer.weights[0].numpy())
    layers.append((Wo, None))

    import pandas as pd
    raw = pd.read_excel(_find("MainData.xlsx"),
                        usecols=["Year", "CountryCode", "TempPopWeight", "PrecipPopWeight"])
    raw = raw[raw.Year <= inst.data_end]
    pT = raw.pivot_table(index="Year", columns="CountryCode", values="TempPopWeight")
    pP = raw.pivot_table(index="Year", columns="CountryCode", values="PrecipPopWeight")
    mT, sT = np.nanmean(pT.values), np.nanstd(pT.values)
    mP, sP = np.nanmean(pP.values), np.nanstd(pP.values)
    y0 = int(min(pT.index))

    sw = lambda z: z / (1.0 + np.exp(-z))
    Tv = np.linspace(0, 30, 90)
    Tz = (Tv - mT) / sT
    Pz = np.full_like(Tv, 0.0)
    dyn = layers[0][0].shape[0] == 3

    out = []
    for yr in (TREF if dyn else [None]):
        cols = [Tz, Pz]
        if dyn:
            t_raw = yr - (y0 - 1)
            t = (t_raw - tmu) / tsig if (tmu is not None and tsig) else t_raw
            cols.append(np.full_like(Tz, t))
        h = np.stack(cols, axis=-1)
        for j, (W, b) in enumerate(layers):
            h = h @ W + (0 if b is None else b)
            if j < len(layers) - 1:
                h = sw(h)
        s = 100 * (h[:, 0] - h[:, 0][int(np.argmin(abs(Tv - mT)))])
        opt = Tv[int(np.argmax(s))]
        out.append((yr, s.max() - s.min(), opt, 0.4 < opt < 29.6))
    return out


def _find(name):
    from utils.miscelaneous.find_data_file import Find_data_file
    return Find_data_file(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node", default="(4,4,4)")
    ap.add_argument("--patience", type=int, default=30, help="SHORT fit; not production")
    ap.add_argument("--min-delta", type=float, default=1e-5)
    ap.add_argument("--verbose", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None,
                    help="override instance.optim.seed_value; the initialiser "
                         "comparison is ONE local optimum per seed, so repeat "
                         "with several before trusting the shape")
    a = ap.parse_args()

    import ast
    from omegaconf import OmegaConf
    from utils.config import flatten_instance
    node = tuple(ast.literal_eval(a.node))

    inst = flatten_instance(OmegaConf.load(ROOT / "config/config.yaml").instance)
    if a.seed is not None:
        inst.seed_value = a.seed
    print("=" * 74)
    print(f"  1 CONFIG   (seed_value = {inst.get('seed_value')})")
    for k in ("data_source", "dynamic_model", "within_projection", "time_scaling",
              "output_initializer", "country_trends", "quadratic_trends", "data_end"):
        print(f"      {k:20s} = {inst.get(k)}")
    print(f"      architecture         = {node}   (short fit: patience={a.patience})")
    if str(inst.get("data_source")).lower() != "wb" or not inst.get("dynamic_model"):
        print("\n      !! config is not wb + dynamic. For the initialiser experiment set")
        print("         instance.model.data_source=wb and instance.model.dynamic_model=true,")
        print("         otherwise you change two things at once.")

    results = {}
    for which in ("zeros", "he_normal"):
        inst.output_initializer = which
        print("\n" + "=" * 74)
        print(f"  fitting with output_initializer = {which}")
        m, info = build_and_fit(inst, node, a.patience, a.min_delta, a.verbose)

        w0 = info["w_out0"]
        is_zero = np.all(w0 == 0.0)
        ok = (which == "zeros") == is_zero
        print(f"    3 INIT       |W_out| at init = {np.abs(w0).max():.4e}  "
              f"({'all zero' if is_zero else 'non-zero'})  "
              f"{'OK' if ok else '*** MISMATCH ***'}")
        if which == "zeros":
            print(f"                 -> dL/dh = 0 on step 1; hidden layers start frozen")
        if info["time_mu"] is not None:
            print(f"    2 TIME SCALE Vectorize reports mu={info['time_mu']:.2f} "
                  f"sigma={info['time_sigma']:.4f}  "
                  f"{'STANDARDISED' if info['time_sigma'] != 1.0 else 'RAW (not standardised!)'}")
        else:
            print("    2 TIME SCALE no 'time' Vectorize layer found (static model?)")
        print(f"    4 SHORT FIT  {info['epochs']} epochs in {info['secs']:.0f}s   "
              f"loss {info['loss']:.6e}   BIC {info['BIC']:.1f}   m_eff {info['m_eff']}")
        for yr, rng, opt, interior in surface_stats(m, inst, info["time_mu"], info["time_sigma"]):
            tag = f"{yr}: " if yr else ""
            print(f"                 {tag}range {rng:6.2f} pp   optimum {opt:5.1f} C"
                  f"   {'interior' if interior else 'AT GRID EDGE'}")
        results[which] = info

    print("\n" + "=" * 74)
    d = results["he_normal"]["BIC"] - results["zeros"]["BIC"]
    print(f"  BIC  zeros {results['zeros']['BIC']:.1f}   he_normal {results['he_normal']['BIC']:.1f}"
          f"   difference {d:+.1f}")
    print("  (short fits - treat as 'does it move at all', not as the answer)")
    print("  static reference on corrected data: BIC -53265.9, range ~19 pp, optimum 17.9 C")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
