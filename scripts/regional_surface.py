"""Shared, TF-free access to the REGIONAL model's per-region response surfaces.

Why this module exists
----------------------
``regional_model/prepare_data.Prepare`` standardises temperature and
precipitation *inside* the per-region loop:

    mean = np.nanmean(pivot_data.values)      # region r only
    std  = np.nanstd(pivot_data.values)
    pivot_data = (pivot_data - mean) / std

so region r's sub-network is trained on inputs expressed in **region r's own**
z-units.  ``Visual_model`` feeds its input straight into ``hidden_1`` and applies
no standardisation of its own.  Evaluating region r's head therefore requires
region r's moments; using the global moments (as the plotting code did before
2026-09-04) evaluates every region at the wrong point of its domain.  For Europe
the two differ by roughly 2.7 standard deviations at 20 C.

``region_moments`` reproduces ``Prepare``'s arithmetic exactly -- same pivot,
same nanmean/nanstd over the pivoted values -- so the z-scores here are the ones
the network actually saw.

Region -> weight-block mapping
------------------------------
``Prepare`` builds its region dicts in the literal order

    Asia, Europe, Africa, Americas, Oceania

``Regions.SetupRegionalModel`` iterates those keys in order, appending one
``BuildRegion`` per region, so Keras names the blocks in the same order: for a
depth-1 architecture, hidden layers ``dense .. dense_4`` and output heads
``dense_5 .. dense_9``.  ``verify_mapping`` checks this against the data instead
of trusting it.

Only depth-1 architectures are supported; the BIC-best regional model is (2,).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from hdf5_min import read_weights  # noqa: E402

# Insertion order of the ``regions`` dict in regional_model/prepare_data.Prepare.
REGION_ORDER = ["Asia", "Europe", "Africa", "Americas", "Oceania"]
# UN M49 codes as used in MainData.RegionCode.
REGION_CODE = {"Asia": 142, "Europe": 150, "Africa": 2, "Americas": 19, "Oceania": 9}

TCOL, PCOL, GCOL = "TempPopWeight", "PrecipPopWeight", "GrowthWDI"


def region_moments(data: pd.DataFrame) -> dict[str, tuple[float, float, float, float]]:
    """Per-region (mean_T, sd_T, mean_P, sd_P), exactly as ``Prepare`` computes them."""
    out = {}
    for region in REGION_ORDER:
        sub = data[data["RegionCode"] == REGION_CODE[region]]
        pt = sub.pivot(index="Year", columns="CountryCode", values=TCOL)
        pp = sub.pivot(index="Year", columns="CountryCode", values=PCOL)
        out[region] = (float(np.nanmean(pt.values)), float(np.nanstd(pt.values)),
                       float(np.nanmean(pp.values)), float(np.nanstd(pp.values)))
    return out


class RegionalSurfaces:
    """Numpy forward pass over the per-region climate sub-networks."""

    def __init__(self, run_dir, node, data):
        node = tuple(node)
        if len(node) != 1:
            raise NotImplementedError(
                f"only depth-1 architectures are supported, got {node}. The layer "
                "naming for deeper regional nets has not been verified."
            )
        self.node = node
        self.run_dir = Path(run_dir)
        self.moments = region_moments(data)
        w = read_weights(str(self.run_dir / "parameters" / f"{node}.weights.h5"))
        n = len(REGION_ORDER)
        self.par = {}
        for k, region in enumerate(REGION_ORDER):
            hid = "layers/dense/vars" if k == 0 else f"layers/dense_{k}/vars"
            self.par[region] = (
                np.asarray(w[f"{hid}/0"], float),                    # kernel (2, node)
                np.asarray(w[f"{hid}/1"], float),                    # bias   (node,)
                np.asarray(w[f"layers/dense_{n + k}/vars/0"], float).ravel(),  # out
            )

    def f(self, region, T, P):
        """Growth response of ``region``'s network at raw (T in C, P in mm)."""
        mT, sT, mP, sP = self.moments[region]
        K, c0, bt = self.par[region]
        Tz, Pz = (np.asarray(T, float) - mT) / sT, (np.asarray(P, float) - mP) / sP
        o = 0.0
        for j in range(K.shape[1]):
            z = c0[j] + K[0, j] * Tz + K[1, j] * Pz
            o = o + bt[j] * (z / (1.0 + np.exp(-z)))
        return o

    # ------------------------------------------------------------------ #
    def verify_mapping(self, data, verbose=True):
        """Check the region -> weight-block assignment against the data.

        For every (block, region) pair, correlate the block's predicted response
        at region r's observed (T, P) with region r's country- and year-demeaned
        growth.  The assumed diagonal assignment should win its own row: a
        network fitted to region r should track region r's within variation
        better than a network fitted to a different climate regime.

        Returns (ok, matrix) with matrix[block, region] = correlation.
        """
        d = data.dropna(subset=[TCOL, PCOL, GCOL])
        M = np.full((len(REGION_ORDER), len(REGION_ORDER)), np.nan)
        for ri, region in enumerate(REGION_ORDER):
            sub = d[d["RegionCode"] == REGION_CODE[region]].copy()
            g = sub[GCOL].to_numpy(float)
            g = (g - sub.groupby("CountryCode")[GCOL].transform("mean").to_numpy(float)
                   - sub.groupby("Year")[GCOL].transform("mean").to_numpy(float)
                   + g.mean())
            T, P = sub[TCOL].to_numpy(float), sub[PCOL].to_numpy(float)
            for bi, block in enumerate(REGION_ORDER):
                # block bi's weights, evaluated in REGION r's z-units
                mT, sT, mP, sP = self.moments[region]
                K, c0, bt = self.par[block]
                Tz, Pz = (T - mT) / sT, (P - mP) / sP
                o = 0.0
                for j in range(K.shape[1]):
                    z = c0[j] + K[0, j] * Tz + K[1, j] * Pz
                    o = o + bt[j] * (z / (1.0 + np.exp(-z)))
                M[bi, ri] = np.corrcoef(o, g)[0, 1] if np.std(o) > 0 else np.nan
        diag_wins = int(sum(np.nanargmax(np.abs(M[:, i])) == i
                            for i in range(len(REGION_ORDER))))
        if verbose:
            print("  |corr| of each block's response with each region's demeaned growth")
            hdr = "block / region"
            print(f"    {hdr:<14}" + "".join(f"{r:>11}" for r in REGION_ORDER))
            for bi, block in enumerate(REGION_ORDER):
                cells = "".join(f"{abs(M[bi, i]):11.3f}" for i in range(len(REGION_ORDER)))
                print(f"    {block:<14}{cells}")
            print(f"    assumed diagonal wins its column in "
                  f"{diag_wins}/{len(REGION_ORDER)} regions")
        return diag_wins, M
