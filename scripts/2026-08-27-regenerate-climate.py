#!/usr/bin/env python3
"""Regenerate population-weighted country climate series from raw grids.

FIXES the aggregation bug: the previous pipeline summed population density over
ALL of a country's cells in the denominator while the numerator silently dropped
cells with missing climate data, scaling affected countries by
alpha = (weight on valid cells)/(total weight). Islands and coastal states were
understated by up to 80%.

Correct formula, applied per month:

    X_it = sum_{j in country i, X_jt VALID} X_jt * d_j
           -------------------------------------------
           sum_{j in country i, X_jt VALID}      d_j

Temperature is then the mean of the 12 monthly means; precipitation is the sum
of the 12 monthly totals (matching the paper's definitions).

Grids (all reprojected to north->south, 0.5 degree, 360x720):
  * CRU TS 4.08 tmp / pre          (NetCDF-3, south->north, flipped here)
  * GPW v4 population density 2000 (NetCDF-3, north->south)
  * GPW v4 national identifier grid (ASCII, north->south) - already on the CRU
    grid, so country assignment is a direct raster lookup with no polygon ops.

Outputs a tidy CSV: CountryCode, Year, TempPopWeight, PrecipPopWeight.

Run from the repo root:  python scripts/2026-08-27-regenerate-climate.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from netcdf3_min import NC3  # noqa: E402

RAW = ROOT / "data/RawData"
Y0, Y1 = 1961, 2023
CRU_START_YEAR = 1901
POP_RASTER = 0            # index 0 = year 2000, as in the paper
OUT = ROOT / "data/climate_country_corrected.csv"


def main():
    # ---- static grids -----------------------------------------------------
    ids = np.loadtxt(RAW / "national-identifier-grid/"
                           "gpw_v4_national_identifier_grid_rev11_30_min.asc",
                     skiprows=6).ravel()

    p = NC3(str(RAW / "gpw_v4_population_density_rev11_30_min.nc"))
    VN = ("Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): "
          "30 arc-minutes")
    fvp = p.fill_value(VN)
    dens = np.asarray(p.read_record(VN, POP_RASTER), float).ravel()
    dens = np.where((dens == fvp) | ~np.isfinite(dens) | (dens < 0), 0.0, dens)

    lut = pd.read_csv(RAW / "national-identifier-grid/"
                            "gpw_v4_national_identifier_grid_rev11_lookup.txt",
                      sep="\t")
    code2iso = dict(zip(lut.Value, lut.ISOCODE))

    keep = (ids >= 0) & (ids < 900) & np.isfinite(ids)
    codes = np.unique(ids[keep]).astype(int)
    cidx = {c: k for k, c in enumerate(codes)}
    cid = np.full(ids.shape, -1, dtype=np.int64)
    for c in codes:
        cid[ids == c] = cidx[c]
    nC = len(codes)
    print(f"grid {ids.size:,} cells | countries {nC} | pop-weighted cells "
          f"{(dens > 0).sum():,}")

    # ---- CRU readers ------------------------------------------------------
    tnc = NC3(str(RAW / "cru_ts4.08.1901.2023.tmp.dat.nc"))
    pnc = NC3(str(RAW / "cru_ts4.08.1901.2023.pre.dat.nc"))
    fvt, fvpre = tnc.fill_value("tmp"), pnc.fill_value("pre")

    years = np.arange(Y0, Y1 + 1)
    T = np.full((len(years), nC), np.nan)
    P = np.full((len(years), nC), np.nan)

    def month_agg(nc, var, fv, rec):
        """Population-weighted country means for one month, valid cells only."""
        x = np.asarray(nc.read_record(var, rec), float)[::-1].ravel()  # -> N->S
        ok = np.isfinite(x) & (x != fv) & (cid >= 0) & (dens > 0)
        num = np.bincount(cid[ok], weights=(x * dens)[ok], minlength=nC)
        den = np.bincount(cid[ok], weights=dens[ok], minlength=nC)
        out = np.full(nC, np.nan)
        nz = den > 0
        out[nz] = num[nz] / den[nz]          # <-- denominator: VALID cells only
        return out

    for i, y in enumerate(years):
        base = (y - CRU_START_YEAR) * 12
        mt = np.vstack([month_agg(tnc, "tmp", fvt, base + m) for m in range(12)])
        mp = np.vstack([month_agg(pnc, "pre", fvpre, base + m) for m in range(12)])
        T[i] = np.nanmean(mt, axis=0)        # annual mean of monthly means
        P[i] = np.nansum(mp, axis=0)         # annual total
        P[i][np.all(np.isnan(mp), axis=0)] = np.nan
        if (y - Y0) % 20 == 0:
            print(f"  {y} done")

    iso = [code2iso.get(c, None) for c in codes]
    rec = []
    for k, c in enumerate(codes):
        if not iso[k]:
            continue
        for i, y in enumerate(years):
            if np.isfinite(T[i, k]):
                rec.append((iso[k], int(y), T[i, k], P[i, k]))
    out = pd.DataFrame(rec, columns=["CountryCode", "Year",
                                     "TempPopWeight", "PrecipPopWeight"])
    out.to_csv(OUT, index=False)
    print(f"\nwrote {OUT}  ({len(out):,} country-years, "
          f"{out.CountryCode.nunique()} countries)")


if __name__ == "__main__":
    main()
