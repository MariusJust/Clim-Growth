"""Bulk download of CMIP6 monthly fields from the Copernicus CDS.

Pulls near-surface air temperature (and optionally precipitation) for a screened
set of models under one or more SSP scenarios, 2015 to 2100, into
data/projections/cmip6/raw/. One NetCDF (or zip) per model x scenario x variable.

    pip install cdsapi
    # put your key in %USERPROFILE%\\.cdsapirc  (see your CDS profile page)

    python scripts/2026-09-28-download-cmip6.py --dry-run     # print requests only
    python scripts/2026-09-28-download-cmip6.py               # core models, ssp585
    python scripts/2026-09-28-download-cmip6.py --set all --scenarios ssp5_8_5 ssp2_4_5
    python scripts/2026-09-28-download-cmip6.py --variables tas pr

Resumable: a file that already exists and is non-empty is skipped, so you can
interrupt and rerun. Failures are logged and do not abort the batch, because
not every model publishes every scenario.

WHY MONTHLY, NOT DAILY
    Only annual means are needed downstream. Monthly is roughly 30x smaller and
    the annual mean of monthly means is exact for temperature (equal-length
    months aside) and for precipitation totals.

WHAT HAPPENS NEXT (not this script)
    These are gridded fields on each model's native grid. They must be
    aggregated to countries with the SAME population weights used to build
    TempPopWeight, otherwise the projection is inconsistent with estimation.
    NB: superseded 2026-11-08. MakeTemp.py carried a denominator bug and an
    AUS hardcode and is NOT the aggregator any more; it now sits in
    src/projections/archive/. Use scripts/2026-09-28-cmip6-country-aggregate.py,
    which fixes both (see its header) and defaults to GPW population weighting
    so the projection inputs match TempPopWeight in estimation.

    Then convert to anomalies against each model's own 2015-2024 baseline and
    add them to the OBSERVED baseline. Raw CMIP6 country temperatures carry
    biases of one to several degrees, and the response surface is curved, so
    absolute model temperatures must not be fed to it directly.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/projections/cmip6/raw"

DATASET = "projections-cmip6"

# CDS model identifiers. VERIFY these against the download form: the API wants
# exact strings and they change occasionally.
#
# CORE: within or near the IPCC AR6 assessed ranges (TCR 1.4-2.2 C,
# ECS 2.5-4.0 C). HOT: repeatedly flagged as exceeding the assessed range; keep
# them for an explicitly labelled upper case, not for the headline. See
# Hausfather et al. on screening, and Zelinka et al. (2020) for per-model values.
# Check each model's TCR yourself before committing to this split in the paper.
MODELS = {
    "core": [
        "bcc_csm2_mr",       # China
        "cmcc_cm2_sr5",      # Italy
        "fgoals_g3",         # China
        "gfdl_esm4",         # USA
        "miroc6",            # Japan
        "mpi_esm1_2_lr",     # Germany
        "mri_esm2_0",        # Japan
        "noresm2_mm",        # Norway
    ],
    "hot": [
        "canesm5",           # Canada
        "cesm2",             # USA
        "cnrm_esm2_1",       # France
        "hadgem3_gc31_ll",   # UK
        "ipsl_cm6a_lr",      # France
        "ukesm1_0_ll",       # UK
    ],
    # Below the assessed range. Included only if you want to bracket from below.
    "cold": [
        "inm_cm4_8",         # Russia
        "inm_cm5_0",         # Russia
    ],
}

VARIABLES = {
    "tas": "near_surface_air_temperature",
    "pr": "precipitation",
}

YEARS = [str(y) for y in range(2015, 2101)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]


def build_request(model: str, scenario: str, variable_key: str) -> dict:
    """The CDS request payload.

    RECONCILE THIS AGAINST THE SNIPPET THE DOWNLOAD FORM GIVES YOU. The form is
    the authoritative source for field names and accepted values; if anything
    below disagrees with it, the form wins.
    """
    return {
        "temporal_resolution": "monthly",
        "experiment": scenario,
        "variable": VARIABLES[variable_key],
        "model": model,
        "year": YEARS,
        "month": MONTHS,
    }


def target_path(model: str, scenario: str, variable_key: str, suffix: str) -> Path:
    return OUT / f"{variable_key}_{scenario}_{model}_2015-2100{suffix}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", dest="mset", default="core",
                    choices=["core", "hot", "cold", "core+hot", "all"],
                    help="which screened model group to download")
    ap.add_argument("--models", nargs="*", default=None,
                    help="explicit model ids, overrides --set")
    ap.add_argument("--scenarios", nargs="*", default=["ssp5_8_5"],
                    help="e.g. ssp1_2_6 ssp2_4_5 ssp5_8_5")
    ap.add_argument("--variables", nargs="*", default=["tas"], choices=list(VARIABLES))
    ap.add_argument("--suffix", default=".zip",
                    help="target extension; CDS returns zip for multi-file results")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the requests and exit without contacting CDS")
    ap.add_argument("--retries", type=int, default=2)
    a = ap.parse_args()

    if a.models:
        models = a.models
    elif a.mset == "all":
        models = MODELS["core"] + MODELS["hot"] + MODELS["cold"]
    elif a.mset == "core+hot":
        models = MODELS["core"] + MODELS["hot"]
    else:
        models = MODELS[a.mset]

    jobs = [(m, s, v) for v in a.variables for s in a.scenarios for m in models]

    print(f"dataset   {DATASET}")
    print(f"models    {len(models)}: {', '.join(models)}")
    print(f"scenarios {', '.join(a.scenarios)}")
    print(f"variables {', '.join(a.variables)}")
    print(f"years     {YEARS[0]}-{YEARS[-1]} monthly")
    print(f"target    {OUT}")
    print(f"jobs      {len(jobs)}\n")

    if a.dry_run:
        for m, s, v in jobs:
            print(f"--- {target_path(m, s, v, a.suffix).name}")
            print(json.dumps(build_request(m, s, v) | {"year": f"[{YEARS[0]}..{YEARS[-1]}]",
                                                       "month": "[01..12]"}, indent=2))
        print("\ndry run: nothing downloaded")
        return 0

    try:
        import cdsapi
    except ImportError:
        print("cdsapi is not installed.  pip install cdsapi")
        return 1

    OUT.mkdir(parents=True, exist_ok=True)
    client = cdsapi.Client()

    manifest, failed = [], []
    for i, (m, s, v) in enumerate(jobs, 1):
        dest = target_path(m, s, v, a.suffix)
        tag = f"[{i}/{len(jobs)}] {dest.name}"

        if dest.exists() and dest.stat().st_size > 0:
            print(f"{tag}  SKIP (exists, {dest.stat().st_size/1e6:.1f} MB)")
            manifest.append(dict(model=m, scenario=s, variable=v,
                                 file=dest.name, status="skipped",
                                 bytes=dest.stat().st_size))
            continue

        req = build_request(m, s, v)
        for attempt in range(1, a.retries + 2):
            try:
                print(f"{tag}  requesting (attempt {attempt})")
                t0 = time.time()
                client.retrieve(DATASET, req).download(str(dest))
                mb, dt = dest.stat().st_size / 1e6, time.time() - t0
                print(f"{tag}  OK {mb:.1f} MB in {dt/60:.1f} min")
                manifest.append(dict(model=m, scenario=s, variable=v,
                                     file=dest.name, status="downloaded",
                                     bytes=dest.stat().st_size))
                break
            except Exception as exc:                                # noqa: BLE001
                print(f"{tag}  FAILED: {type(exc).__name__}: {exc}")
                if dest.exists():
                    dest.unlink(missing_ok=True)
                if attempt == a.retries + 1:
                    failed.append(dict(model=m, scenario=s, variable=v,
                                       error=f"{type(exc).__name__}: {exc}"))
                else:
                    time.sleep(10 * attempt)

    mf = OUT / "manifest.json"
    mf.write_text(json.dumps(dict(
        generated=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        dataset=DATASET, scenarios=a.scenarios, variables=a.variables,
        requested=len(jobs), ok=len(manifest), failed=failed,
        entries=manifest), indent=2), encoding="utf-8")

    print(f"\n{len(manifest)} of {len(jobs)} present, {len(failed)} failed")
    for f in failed:
        print(f"   {f['variable']} {f['scenario']} {f['model']}: {f['error']}")
    print(f"manifest -> {mf}")
    print("\nNot every model publishes every scenario. Failures are expected for "
          "some combinations; check the CDS form before assuming a bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
