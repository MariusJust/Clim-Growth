"""Queue the three holiday jobs in one command.

    python scripts/2026-09-08-dispatch-holiday-jobs.py            # submit
    python scripts/2026-09-08-dispatch-holiday-jobs.py --dry-run  # print only

Jobs, in order:

  J1  levels rerun, output_initializer = zeros
      Parity with the growth headline run, which used the code default (zeros).
      The 2026-09-04 levels run used he_normal, so the two targets were optimised
      differently. Architecture set is left at the levels run's 54 so J1 vs
      2026-09-04 is a clean zeros-against-he_normal comparison.

  J2  dynamic rerun, dynamic_include_time = true
      The like-for-like test PROJECT.md D12 asks for. Retaining the additive year
      FE puts the dynamic model on the same nuisance basis as the static one and
      forces t to enter only through its interaction with (T, P).

  J3  bootstrap of J1, held with --dependency=afterok until J1 succeeds
      Gives the levels projections confidence bands. J1's run directory is
      created by slurm.py at submit time, so it can be wired in before J1 runs.

slurm.py keeps its own sequential queue in runs/estimation/.last_job_id and
chains each new estimation job after the previous one, so J1 and J2 serialise
automatically. J3 is chained explicitly on J1.

How it works: config/config.yaml is the single source of truth for slurm.py, so
this script patches it per job, submits, and restores it. The original is backed
up first and restored in a finally block, so an interrupted run cannot leave the
config in a job-specific state.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config/config.yaml"
BACKUP = ROOT / f"config/config.yaml.bak-{datetime.now():%Y%m%d-%H%M%S}"
DRY = "--dry-run" in sys.argv

# key -> value, applied to the flat "key: value" lines of config.yaml
J1 = {"target_mode": "'levels'", "output_initializer": "zeros",
      "dynamic_model": "false", "dynamic_include_time": "false"}
J2 = {"target_mode": "'growth'", "output_initializer": "he_normal",
      "dynamic_model": "true", "dynamic_include_time": "true"}


def patch(mapping: dict[str, str]) -> None:
    """Rewrite the given top-level model keys in place, preserving indentation."""
    text = CONFIG.read_text(encoding="utf-8")
    for key, val in mapping.items():
        pat = re.compile(rf"^(\s*){key}:\s*[^#\n]*(#.*)?$", re.MULTILINE)
        if not pat.search(text):
            raise SystemExit(f"key '{key}' not found in {CONFIG}; aborting")
        text = pat.sub(lambda m: f"{m.group(1)}{key}: {val}"
                       + (f"   {m.group(2)}" if m.group(2) else ""), text, count=1)
    CONFIG.write_text(text, encoding="utf-8")


def show(label: str, mapping: dict[str, str]) -> None:
    print(f"\n=== {label} ===")
    for k, v in mapping.items():
        print(f"    {k}: {v}")


def submit(script: str, *extra: str) -> tuple[str, str]:
    """Run a dispatch script; return (stdout, job id) with '' when unknown."""
    cmd = [sys.executable, str(ROOT / "scripts" / script), *extra]
    print(f"    $ {' '.join(cmd)}")
    if DRY:
        return "", ""
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    if r.returncode != 0:
        raise SystemExit(f"{script} failed with code {r.returncode}; "
                         f"config restored, nothing further submitted")
    # slurm.py prints "Submitted job 12345: name"; slurm_bootstrap.py prints
    # "submitted job 12345". Match either.
    m = re.search(r"submitted job (\d+)", r.stdout, re.IGNORECASE)
    return r.stdout, (m.group(1) if m else "")


def newest_estimation_run() -> str:
    """slurm.py creates its run directory at submit time but does not print it,
    so take the most recently created directory under runs/estimation/."""
    d = ROOT / "runs/estimation"
    dirs = [p for p in d.iterdir() if p.is_dir()]
    if not dirs:
        return ""
    newest = max(dirs, key=lambda p: p.stat().st_mtime)
    return str(newest.relative_to(ROOT)).replace("\\", "/")


def main() -> None:
    if not CONFIG.exists():
        raise SystemExit(f"missing {CONFIG}")
    shutil.copy2(CONFIG, BACKUP)
    print(f"config backed up -> {BACKUP.name}")
    if DRY:
        print("DRY RUN: config will be patched and restored, nothing submitted")

    try:
        show("J1  levels rerun, output_initializer=zeros", J1)
        patch(J1)
        out1, id1 = submit("slurm.py")
        rd1 = "" if DRY else newest_estimation_run()
        print(f"    J1 job id {id1 or '(unknown)'}   run dir {rd1 or '(unknown)'}")
        if not DRY and "_levels" not in rd1:
            raise SystemExit(
                f"expected J1's run dir to end in '_levels', got '{rd1}'. "
                "Refusing to chain the bootstrap onto the wrong run; config restored."
            )

        show("J2  dynamic rerun, dynamic_include_time=true", J2)
        patch(J2)
        out2, id2 = submit("slurm.py")
        print(f"    J2 job id {id2 or '(unknown)'}")

        show("J3  levels bootstrap, held on J1", {"bootstrap.run_dir": rd1 or "<J1 run dir>",
                                                  "dependency": f"afterok:{id1 or '<J1 id>'}"})
        if not DRY and not (rd1 and id1):
            print("    SKIPPED: could not read J1's job id or run dir from slurm.py output.")
            print("    Submit it by hand once J1 is running:")
            print("      set instance.bootstrap.run_dir to J1's run dir in config.yaml, then")
            print("      python scripts/slurm_bootstrap.py --after <J1 job id>")
        else:
            patch({**J1})
            text = CONFIG.read_text(encoding="utf-8")
            text = re.sub(r"^(\s*)run_dir:.*$",
                          lambda m: f"{m.group(1)}run_dir: {rd1 or '<J1 run dir>'}",
                          text, count=1, flags=re.MULTILINE)
            CONFIG.write_text(text, encoding="utf-8")
            submit("slurm_bootstrap.py", "--after", id1 or "0")
    finally:
        shutil.copy2(BACKUP, CONFIG)
        print(f"\nconfig restored from {BACKUP.name}")
        print("check the queue with:  squeue -u $USER")


if __name__ == "__main__":
    main()
