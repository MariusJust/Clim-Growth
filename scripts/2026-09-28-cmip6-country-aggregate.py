"""Population-weighted country temperature from CMIP6, and the model ensemble.

Replaces src/projections/MakeTemp.py, which cannot be used as it stands. Three
bugs there, all verified on 2026-09-28:

  1. AUS hardcode. Inside `for iso, geometry in country_codes.items()` the loop
     variable is reassigned (`iso="AUS"`) BEFORE it is used as the output key, so
     every country receives Australia's geometry and the output dict ends up with
     a single entry.
  2. Denominator mismatch. `np.nansum(tmp[mask]*w) / np.nansum(w)` drops NaN
     cells from the numerator while keeping their weights in the denominator,
     biasing every country mean toward zero. Same class as the bug behind the
     Burke discrepancy.
  3. No area term. GPW density is persons per km2 of LAND, so population is
     density x land_area. Using density alone over-weights small high-latitude
     cells. Verified: density_2020 x land_area sums to 7.80 bn, matching UN 2020.

DESIGN
Everything is done on the fixed GPW 30 arc-minute grid (720 x 360, -180..180,
-90..90). Weights are therefore built ONCE and are identical for every model;
only the interpolation of each model onto that grid is model specific. This
removes the per-model polygon masking in MakeTemp.py, which silently dropped
small countries whenever no grid-cell centre fell inside them, with the set of
dropped countries differing by model grid.

    python scripts/2026-09-28-cmip6-country-aggregate.py --weights-only
    python scripts/2026-09-28-cmip6-country-aggregate.py
    python scripts/2026-09-28-cmip6-country-aggregate.py --no-fill

Outputs, in data/projections/cmip6/:
    weights_30min.npz                  country id and population weight per cell
    country_tas_by_model_ssp585.csv    model, iso, year, tas_C          (absolute)
    country_tas_ensemble_ssp585.csv    iso, year, tas_C, delta_C, n_models
                                       delta_C is the anomaly against each
                                       model's own 2015-2024 mean, ensemble
                                       median. USE delta_C, not tas_C.

WHY DELTAS
Raw CMIP6 country temperatures carry biases of one to several degrees. The
response surface is curved and the result turns on where a country sits relative
to its optimum, so absolute model temperatures must never be fed to it. Add
delta_C to the OBSERVED baseline instead.
"""
from __future__ import annotations

import argparse
import glob
import struct
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/projections/cmip6/raw"
OUT = ROOT / "data/projections/cmip6"
GPW = ROOT / "data/RawData"
NID = GPW / "national-identifier-grid/gpw_v4_national_identifier_grid_rev11_30_min.asc"
LOOK = GPW / "national-identifier-grid/gpw_v4_national_identifier_grid_rev11_lookup.txt"
POP = GPW / "gpw_v4_population_density_rev11_30_min.nc"

BAND_DENSITY_2020 = 4      # bands 0-4 are density 2000, 2005, 2010, 2015, 2020
BAND_LAND_AREA = 8         # verified: density_2020 x this sums to 7.80 bn
BASE0, BASE1 = 2015, 2024  # baseline window for the delta method


# ---------------------------------------------------------------------------
# minimal netCDF-3 classic reader (the GPW file is classic; no netCDF4 needed)
# ---------------------------------------------------------------------------
_SZ = {1: 1, 2: 1, 3: 2, 4: 4, 5: 4, 6: 8}
_DT = {1: "i1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8"}


class _Reader:
    def __init__(self, b):
        self.b, self.p = b, 0

    def u32(self):
        v = struct.unpack_from(">I", self.b, self.p)[0]
        self.p += 4
        return v

    def name(self):
        n = self.u32()
        s = self.b[self.p:self.p + n]
        self.p += n + ((4 - n % 4) % 4)
        return s.decode("utf8", "replace")

    def vals(self, t, n):
        c = _SZ[t] * n
        raw = self.b[self.p:self.p + c]
        self.p += c + ((4 - c % 4) % 4)
        return raw.decode("utf8", "replace") if t == 2 else np.frombuffer(raw, dtype=_DT[t])

    def atts(self):
        tag, n = self.u32(), self.u32()
        out = {}
        if tag == 0 and n == 0:
            return out
        for _ in range(n):
            k = self.name()
            t, ln = self.u32(), self.u32()
            out[k] = self.vals(t, ln)
        return out


def read_nc3(path):
    raw = Path(path).read_bytes()
    ver = raw[3]
    r = _Reader(raw)
    r.p = 4
    r.u32()
    tag, nd = r.u32(), r.u32()
    dims = []
    if tag != 0:
        for _ in range(nd):
            dims.append((r.name(), r.u32()))
    r.atts()
    tag, nv = r.u32(), r.u32()
    vars_ = {}
    if tag != 0:
        for _ in range(nv):
            nm = r.name()
            k = r.u32()
            ids = [r.u32() for _ in range(k)]
            va = r.atts()
            t, vsize = r.u32(), r.u32()
            beg = struct.unpack_from(">Q", raw, r.p)[0] if ver == 2 else r.u32()
            if ver == 2:
                r.p += 8
            vars_[nm] = dict(type=t, dims=[dims[i][0] for i in ids],
                             begin=beg, vsize=vsize, atts=va)
    return raw, dict(dims), vars_


# ---------------------------------------------------------------------------
# weights on the fixed 0.5 degree grid
# ---------------------------------------------------------------------------
def build_weights(fill_water: bool = True, fill_iters: int = 2, weighting: str = "area"):
    """Per-cell country id and weight on the fixed 0.5 degree grid.

    weighting='area'       cos(lat) cell area. No GPW population involved, so
                           none of the density gaps below can bite. Every country
                           in the identifier grid gets a value.
    weighting='population' GPW 2020 density x land area. More faithful to how
                           people actually experience climate, but the density and
                           land-area layers have gaps over small and coastal
                           states and need the fallbacks documented below.
    """
    with open(NID) as f:
        hdr = {}
        for _ in range(6):
            k, v = f.readline().split()
            hdr[k] = float(v)
        nid = np.loadtxt(f).astype(np.int32)
    assert nid.shape == (360, 720), nid.shape

    # ---- area weighting: identifier grid + geometry only -------------------
    look = pd.read_csv(LOOK, sep="\t")
    code2iso = dict(zip(look.Value.astype(int), look.ISOCODE.astype(str)))
    if weighting == "area":
        nlat, nlon = 360, 720
        lat = 89.75 - np.arange(nlat) * 0.5
        lon = -179.75 + np.arange(nlon) * 0.5
        cell = (np.deg2rad(0.5) * 6371.0) ** 2 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, nlon))
        valid = (nid > 0) & (nid != 32767)
        iso = np.full(nid.shape, "", dtype=object)
        for c in np.unique(nid[valid]):
            iso[nid == c] = code2iso.get(int(c), "")
        w = np.where(valid, cell, 0.0)
        ncty = len({s for s in np.unique(iso) if s})
        print(f"  weighting       AREA (cos-lat cell area); no GPW population used")
        print(f"  grid            {nid.shape}, lat {lat[0]:.2f}..{lat[-1]:.2f}, "
              f"lon {lon[0]:.2f}..{lon[-1]:.2f}")
        print(f"  countries       {ncty} with at least one cell")
        pd.DataFrame([dict(iso=code2iso.get(int(c), ""), cells=int((nid == c).sum()),
                           area_km2=float(cell[nid == c].sum()))
                      for c in np.unique(nid[valid])]).sort_values(
            "area_km2", ascending=False).to_csv(OUT / "weights_country_coverage.csv", index=False)
        return dict(iso=iso, weight=w, lat=lat, lon=lon, nid=nid)

    raw, dims, V = read_nc3(POP)
    nlat, nlon = dims["latitude"], dims["longitude"]
    lat = np.frombuffer(raw, ">f8", count=nlat, offset=V["latitude"]["begin"])
    lon = np.frombuffer(raw, ">f8", count=nlon, offset=V["longitude"]["begin"])
    pv = next(k for k in V if k.startswith("Population Density"))
    rec = V[pv]["vsize"]

    def band(b):
        a = np.frombuffer(raw, ">f4", count=nlat * nlon,
                          offset=V[pv]["begin"] + b * rec).reshape(nlat, nlon)
        return np.where(a > -1e30, a, np.nan)

    # Population = density x land area. Band 8 carries land area and validates
    # against UN 2020 totals for large countries, but it is NaN over small and
    # narrow coastal states: Portugal has 8 cells with finite density and ZERO
    # finite land area, so the bare product silently zeroed 30 countries
    # including PRT, CRI, SLV, LBN, JAM. Fall back to geometric cell area
    # (R^2 dlon dlat cos(lat), which sums to 510.1e6 km2, the Earth's surface)
    # wherever band 8 is absent. That overstates partly-ocean cells, but the
    # weights only ever enter as within-country ratios, so the effect is second
    # order, and no country can be dropped.
    dens = band(BAND_DENSITY_2020)
    land = band(BAND_LAND_AREA)
    cell = (np.deg2rad(0.5) * 6371.0) ** 2 * np.cos(np.deg2rad(lat))[:, None] * np.ones((1, nlon))
    # Scale the fallback by the typical land fraction rather than using the full
    # cell. Bare cell area inflates the fallback cells, which are coastal by
    # construction, and that distorts WITHIN-country weighting for any country
    # holding both cell types (Indonesia reached 1.50x its UN total that way).
    have = np.isfinite(land) & (cell > 0)
    frac = float(np.median(land[have] / cell[have]))
    n_fb = int((np.isfinite(dens) & ~np.isfinite(land)).sum())
    land = np.where(np.isfinite(land), land, cell * frac)
    pop = np.nan_to_num(dens) * land
    print(f"  land area       band {BAND_LAND_AREA} where present; fallback "
          f"cell_area x {frac:.3f} for {n_fb} populated cells")

    valid = (nid > 0) & (nid != 32767)
    covered = pop[valid].sum()
    total = pop.sum()
    print(f"  grid            {nid.shape}, lat {lat[0]:.2f}..{lat[-1]:.2f}, "
          f"lon {lon[0]:.2f}..{lon[-1]:.2f}")
    print(f"  world pop       {total/1e9:.2f} bn (UN 2020 ~7.8)")
    print(f"  on country cells{covered/1e9:.2f} bn  ({100*covered/total:.1f}% coverage)")

    if fill_water:
        # Populated cells the identifier grid calls water are reassigned to the
        # nearest country cell. Without this, dense coastal cells are dropped and
        # country means are pulled toward interior climate; coverage loss is not
        # uniform across countries, so it would distort cross-country comparisons.
        # Iterative 4-neighbour dilation: country ids spread outward from the
        # coast into adjacent populated cells. Cheap (O(iterations x grid)) and
        # geographically the right operation, since the cells being recovered are
        # coastal fringes of the country next to them. Longitude wraps at the
        # antimeridian via np.roll, which is correct on a globe.
        # Iteration count is bounded deliberately. Filled cells become sources on
        # the next pass, so an unbounded loop marches ids across open water: at 5
        # iterations Brazil reaches 1.79x its UN 2020 population by claiming
        # populated Atlantic cells. Measured trade-off (coverage, worst ratio):
        #   0 iters 79.4%, JPN 0.37   2 iters 92.4%, all <= 1.0
        #   3 iters 95.9%, BRA 1.24   5 iters 99.7%, BRA 1.79
        # Two is the largest count at which no country is over-assigned.
        n0 = int(((~valid) & (pop > 0)).sum())
        for _ in range(fill_iters):
            miss = (~valid) & (pop > 0)
            if not miss.any():
                break
            src = np.where(valid, nid, 0)
            filled = np.zeros_like(nid)
            for shift, axis in ((1, 0), (-1, 0), (1, 1), (-1, 1)):
                s = np.roll(src, shift, axis=axis)
                take = miss & (filled == 0) & (s > 0)
                filled[take] = s[take]
            newly = miss & (filled > 0)
            if not newly.any():
                break
            nid[newly] = filled[newly]
            valid = (nid > 0) & (nid != 32767)
        left = int(((~valid) & (pop > 0)).sum())
        print(f"  water fill      recovered {n0-left} of {n0} populated water cells, "
              f"coverage now {100*pop[valid].sum()/total:.1f}%")

    # per-country coverage, so poorly covered countries are visible rather than
    # silently distorting the weighted mean (PAK sits near 0.43 of its UN total
    # at every iteration count: a fixed property of the GPW identifier grid).
    rep = []
    for c in np.unique(nid[valid]):
        m = nid == c
        rep.append(dict(code=int(c), iso=code2iso.get(int(c), ""),
                        cells=int(m.sum()), pop_m=pop[m].sum() / 1e6))
    pd.DataFrame(rep).sort_values("pop_m", ascending=False).to_csv(
        OUT / "weights_country_coverage.csv", index=False)
    print(f"  coverage report {OUT/'weights_country_coverage.csv'}")

    iso = np.full(nid.shape, "", dtype=object)
    for c in np.unique(nid[valid]):
        iso[nid == c] = code2iso.get(int(c), "")
    w = np.where(valid, pop, 0.0)

    # Some countries have cells in the identifier grid but no recorded density in
    # any of them (LBN, BRN, GMB and nine small island states). They would carry
    # zero weight and vanish. Fall back to AREA weighting over their own cells:
    # for a country spanning a handful of 0.5 degree cells, the area mean and the
    # population-weighted mean are close, and a country present in the panel must
    # not be dropped from the projection. D16 makes the median country the
    # headline statistic, so a silently shrinking country set moves the result.
    n_area = 0
    for c in np.unique(nid[valid]):
        m = nid == c
        if w[m].sum() <= 0:
            w[m] = cell[m]
            n_area += 1
    if n_area:
        print(f"  area fallback   {n_area} countries had no recorded population; "
              f"weighted by cell area instead")
    return dict(iso=iso, weight=w, lat=lat, lon=lon, nid=nid)


# ---------------------------------------------------------------------------
# model reading
# ---------------------------------------------------------------------------
def open_model(zpath):
    """Concatenated (time, lat, lon) DataArray in degC on -180..180 longitudes."""
    try:
        import xarray as xr
    except ImportError:
        sys.exit("xarray is required to read the CMIP6 files.  pip install xarray netCDF4")
    import shutil
    import tempfile

    # Windows: xarray keeps the file handle open after .load(), so letting
    # TemporaryDirectory clean up while a dataset is still open raises
    # PermissionError [WinError 32] and loses the model. Each file is therefore
    # opened, loaded, and CLOSED before the directory is removed, and cleanup
    # failures are non-fatal. This also avoids open_mfdataset, whose lazy dask
    # handles have the same problem.
    def _decode_kwargs():
        coder = getattr(getattr(xr, "coders", None), "CFDatetimeCoder", None)
        return {"decode_times": coder(use_cftime=True)} if coder else {"use_cftime": True}

    zf = zipfile.ZipFile(zpath)
    ncs = sorted(n for n in zf.namelist() if n.endswith(".nc"))
    td = tempfile.mkdtemp(prefix="cmip6_")
    try:
        parts, units = [], "K"
        for n in ncs:
            p = zf.extract(n, td)
            ds = xr.open_dataset(p, **_decode_kwargs())
            try:
                v = ds["tas"].load().copy(deep=True)
                units = v.attrs.get("units", units)   # concat can drop attrs
                parts.append(v)
            finally:
                ds.close()
        da = xr.concat(parts, dim="time").sortby("time") if len(parts) > 1 else parts[0]
    finally:
        zf.close()
        shutil.rmtree(td, ignore_errors=True)

    if units == "K":
        da = da - 273.15
    # 0..360 -> -180..180, then sort. Skipping this silently misplaces every
    # Western-hemisphere country.
    if float(da.lon.max()) > 180.0:
        da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
    return da


def to_annual_on_grid(da, lat, lon):
    """Annual means, nearest-neighbour onto the fixed 0.5 degree grid."""
    ann = da.groupby("time.year").mean("time")
    return ann.interp(lat=("lat", lat), lon=("lon", lon), method="nearest")


def country_means(ann, iso, w):
    """Population-weighted country means. Numerator and denominator share a mask,
    which is the fix for MakeTemp.py's denominator bug."""
    isos = sorted({s for s in np.unique(iso) if s})
    years = [int(y) for y in ann.year.values]
    A = ann.values                       # (year, lat, lon)
    rows = []
    for s in isos:
        m = iso == s
        ws = w[m]
        if ws.sum() <= 0:
            continue
        v = A[:, m]
        ok = np.isfinite(v)
        num = np.nansum(np.where(ok, v, 0.0) * ws[None, :], axis=1)
        den = (ok * ws[None, :]).sum(axis=1)
        rows.append(pd.DataFrame(dict(iso=s, year=years,
                                      tas_C=np.where(den > 0, num / np.maximum(den, 1e-12), np.nan))))
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights-only", action="store_true")
    ap.add_argument("--weighting", choices=["area", "population"], default="population",
                    help="GPW population weighting (default) or cos-lat area. "
                         "Population matches TempPopWeight in MainData, so the "
                         "projection inputs are consistent with the response "
                         "surface they are fed into; area was a first-run "
                         "simplification and leaves them inconsistent.")
    ap.add_argument("--scenarios", nargs="*",
                    default=["ssp1_2_6", "ssp2_4_5", "ssp5_8_5"],
                    help="raw-file scenario tags to aggregate")
    ap.add_argument("--fill-iters", type=int, default=2,
                    help="dilation passes; >2 over-assigns across water")
    ap.add_argument("--no-fill", action="store_true",
                    help="do not reassign populated water cells to the nearest country")
    a = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print("=== weights, GPW 30 arc-minute ===")
    W = build_weights(fill_water=not a.no_fill, fill_iters=a.fill_iters,
                      weighting=a.weighting)
    np.savez_compressed(OUT / "weights_30min.npz",
                        iso=W["iso"].astype("U3"), weight=W["weight"],
                        lat=W["lat"], lon=W["lon"])
    print(f"  -> {OUT/'weights_30min.npz'}")
    if a.weights_only:
        return 0

    # Short tags keep the existing ssp585 filenames valid, so nothing downstream
    # that already reads country_tas_ensemble_ssp585.csv has to change.
    short = {"ssp1_2_6": "ssp126", "ssp2_4_5": "ssp245", "ssp5_8_5": "ssp585"}

    # Precipitation is deliberately NOT aggregated. The projections hold P at its
    # observed country baseline; pr_*.zip is downloaded and ready when that
    # changes, but aggregating it needs a flux-to-mm/year conversion with month
    # lengths and a multiplicative anomaly, neither of which is implemented here.

    written = []
    for scen in a.scenarios:
        zips = sorted(glob.glob(str(RAW / f"tas_{scen}_*.zip")))
        if not zips:
            print(f"\n=== {scen}: no model archives in {RAW}, skipped ===")
            continue
        print(f"\n=== {scen}: {len(zips)} models ===")
        frames = []
        for z in zips:
            tag = Path(z).stem.replace(f"tas_{scen}_", "").replace("_2015-2100", "")
            try:
                da = open_model(z)
                ann = to_annual_on_grid(da, W["lat"], W["lon"])
                df = country_means(ann, W["iso"], W["weight"])
                df.insert(0, "model", tag)
                frames.append(df)
                yr = df.groupby("year").tas_C.mean()
                print(f"  {tag:18} {df.iso.nunique():>4} countries  "
                      f"{int(df.year.min())}-{int(df.year.max())}  "
                      f"unweighted mean {yr.iloc[0]:+.2f} -> {yr.iloc[-1]:+.2f} C")
            except Exception as exc:                                # noqa: BLE001
                print(f"  {tag:18} FAILED: {type(exc).__name__}: {exc}")

        if not frames:
            print(f"  {scen}: no model produced output, skipped")
            continue

        sid = short.get(scen, scen)
        allm = pd.concat(frames, ignore_index=True)
        by_model = OUT / f"country_tas_by_model_{sid}.csv"
        allm.to_csv(by_model, index=False)

        # delta method: anomaly against each model's OWN baseline, then ensemble median
        base = (allm[allm.year.between(BASE0, BASE1)]
                .groupby(["model", "iso"]).tas_C.mean().rename("base"))
        d = allm.join(base, on=["model", "iso"])
        d["delta_C"] = d.tas_C - d.base
        ens = (d.groupby(["iso", "year"])
                 .agg(tas_C=("tas_C", "median"), delta_C=("delta_C", "median"),
                      delta_sd=("delta_C", "std"), n_models=("model", "nunique"))
                 .reset_index())
        ensemble = OUT / f"country_tas_ensemble_{sid}.csv"
        ens.to_csv(ensemble, index=False)

        w2100 = ens[ens.year == 2100].delta_C
        print(f"  ensemble: {ens.iso.nunique()} countries, {ens.n_models.max()} models")
        print(f"  warming 2100 vs {BASE0}-{BASE1}: median {w2100.median():.2f} C, "
              f"range {w2100.min():.2f} to {w2100.max():.2f} C")
        written += [by_model, ensemble]

    if not written:
        sys.exit("no scenario produced output")

    print(f"\nweighting {a.weighting}")
    for p in written:
        print(f"  -> {p}")
    print("\nUse delta_C, added to the OBSERVED 2015-2024 country mean. "
          "Absolute tas_C carries model bias and must not enter the response surface.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
