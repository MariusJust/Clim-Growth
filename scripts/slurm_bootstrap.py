"""Generate and submit the SLURM job for the country-cluster bootstrap.

Mirrors scripts/slurm.py, but points at src/bootstrap.py and writes into
runs/bootstrap/<job_id>/ instead of runs/estimation/.

    python scripts/slurm_bootstrap.py

Sizing note. The old figure of 10.6 min per initialisation was measured at
patience=200. Post-D19 the config runs patience=600 with lr_reduce_patience=50,
and n_inits is 3 rather than 2, so there are 3000 fits and each runs longer.
Budget roughly 3x the old estimate, 18-24 h on 55 cores. --time is 48 h.

Run a small n_boot first (25) and read the per-fit time out of the log before
committing: there is no runtime history, bootstrap.log is empty in every
previous run.

The formulation is taken from instance.model.formulation. regional and income
resample WITHIN each group and hold each group's reference country fixed; see
resample_columns in src/bootstrap.py.
"""
import argparse
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

ram_allocated = 200

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
RUNS_DIR = ROOT / "runs"

# Deliberately NOT importing anything from src/. utils/__init__.py imports
# visualise, which imports TensorFlow, so `from utils.config import
# flatten_instance` loaded the whole of TF on the login node just to read five
# config values. This launcher only writes a job file and calls sbatch; the
# compute belongs on the compute node.
ap = argparse.ArgumentParser(description="Queue a bootstrap as a SLURM job.")
ap.add_argument("--run-dir", help="estimation run to bootstrap; overrides the config")
ap.add_argument("--formulation", choices=["global", "regional", "income"])
ap.add_argument("--node", help="architecture, or 'auto' to take the best from results.npy")
ap.add_argument("--n-boot", type=int)
ap.add_argument("--after", help="hold until this job id finishes SUCCESSFULLY (afterok)")
ap.add_argument("--after-any", help="hold until this job id finishes, pass or fail (afterany). "
                                    "Right for serialising independent arms.")
args = ap.parse_args()

inst = OmegaConf.load(CONFIG_DIR / "config.yaml").instance
boot = inst.bootstrap
model = inst.model

if args.run_dir:
    boot.run_dir = args.run_dir
if args.n_boot:
    boot.n_boot = args.n_boot
if args.node:
    boot.node = None if args.node == "auto" else args.node

# Inherit the ESTIMATION settings from the run being bootstrapped. Without this
# the bootstrap can silently fit a different model than the run whose band it
# claims to be: the income run has within_projection=false while config.yaml
# says true, so the config alone would build explicit FE dummies for one arm and
# project them out for another.
INHERIT = ("formulation", "within_projection", "country_trends",
           "quadratic_trends", "target_mode", "data_source", "data_end")
overrides = []
run_cfg_path = None
for cand in (Path(boot.run_dir) / ".hydra/config.yaml",
             Path(boot.run_dir) / "config/config.yaml"):
    if (ROOT / cand).exists() or cand.exists():
        run_cfg_path = cand if cand.exists() else ROOT / cand
        break

if run_cfg_path is not None:
    rc = OmegaConf.load(run_cfg_path)
    rc = rc.instance if "instance" in rc else rc
    rm = rc.model if "model" in rc else rc
    print(f"inheriting estimation settings from {run_cfg_path}")
    for k in INHERIT:
        if k in rm:
            v = rm[k]
            if k in model and model[k] != v:
                print(f"  {k}: config says {model[k]}, run says {v} -> using the run's")
            model[k] = v
            overrides.append(f"instance.model.{k}={v}")
else:
    print(f"WARNING: no config found under {boot.run_dir}; "
          f"using config.yaml as-is, so verify formulation and within_projection by hand")

if args.formulation:
    model.formulation = args.formulation
    overrides = [o for o in overrides if not o.startswith("instance.model.formulation=")]
    overrides.append(f"instance.model.formulation={args.formulation}")

node_tag = (str(boot.node).replace("(", "").replace(")", "")
            .replace(",", "-").strip("-") if boot.node else "auto")
job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id += f"_boot{boot.n_boot}x{boot.n_inits}_{model.formulation}_node{node_tag}"
if model.country_trends:
    job_id += "_country_trends"

job_dir = RUNS_DIR / "bootstrap" / job_id
job_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(CONFIG_DIR, job_dir / "config")


def job_func(job_name, job_dir, root_dir, extra):
    s = '#!/bin/bash\n'
    s += f'#SBATCH --job-name={job_name}\n'
    s += '#SBATCH --partition=q64\n'
    s += f'#SBATCH --mem={ram_allocated}G\n'
    s += '#SBATCH --nodes=1\n'
    s += '#SBATCH --time=48:00:00\n'
    s += '#SBATCH --ntasks-per-node=1\n'
    s += '#SBATCH --cpus-per-task=55\n'
    s += f"#SBATCH --output={job_dir / 'slurm-%j.out'}\n"
    s += f"#SBATCH --error={job_dir / 'slurm-%j.err'}\n"
    s += f'cd "{root_dir}"\n'
    s += 'export TF_ENABLE_ONEDNN_OPTS=0\n'
    s += 'export TF_CPP_MIN_LOG_LEVEL=3\n'
    s += 'export CUDA_VISIBLE_DEVICES=""\n'
    # One thread per worker. 55 TensorFlow processes each opening a 55-thread
    # pool is what left the 2026-09-28 estimation job at 10-35% CPU and deep in
    # swap until the cgroup OOM killer took workers out.
    s += 'export OMP_NUM_THREADS=1\n'
    s += 'export TF_NUM_INTRAOP_THREADS=1\n'
    s += 'export TF_NUM_INTEROP_THREADS=1\n'
    s += 'export OPENBLAS_NUM_THREADS=1\n'
    s += 'export MKL_NUM_THREADS=1\n'
    s += (
        f'python -u src/bootstrap.py '
        f'--config-path "{job_dir / "config"}" '
        f'--config-name config '
        f'hydra.run.dir="{job_dir}" '
        f'instance.bootstrap.n_process=55 '
        f'2>&1 | grep -Ev "All log messages before absl::InitializeLog|Unable to register cuDNN factory|Unable to register cuBLAS factory"\n'
    )
    return s


name = f"BOOT_{job_id}"
job_script = job_dir / "slurm.job"
with open(job_script, "w") as f:
    f.write(job_func(name, job_dir, ROOT, overrides))

print(f"n_boot={boot.n_boot}  n_inits={boot.n_inits}  node={boot.node}")
print(f"formulation={model.formulation}  within_projection={model.within_projection}")
print(f"estimation run: {boot.run_dir}")
print(f"job dir: {job_dir}")

# There is no bash fallback: this job must be dispatched to the scheduler, never
# run on the login node.
if shutil.which("sbatch") is None:
    raise SystemExit("sbatch not found. Submit from a SLURM login node.")

submit_cmd = ["sbatch", "--parsable"]

# Optional dependency: --after <jobid> holds this job until <jobid> completes
# successfully. Used to chain a bootstrap behind the estimation run whose
# run_dir it consumes, so the pair can be queued in one sitting.
# afterok: the dependent job is cancelled if the first one FAILS. Right when the
# second job consumes the first one's output.
# afterany: runs once the first finishes either way. Right for merely serialising
# arms that are independent of each other, which the three bootstraps are: a
# failed global should not strand the regional and income jobs in
# DependencyNeverSatisfied forever.
if args.after:
    submit_cmd.append(f"--dependency=afterok:{args.after.strip()}")
    print(f"holding until job {args.after.strip()} SUCCEEDS")
elif args.after_any:
    submit_cmd.append(f"--dependency=afterany:{args.after_any.strip()}")
    print(f"holding until job {args.after_any.strip()} finishes (pass or fail)")

submit_cmd.append(str(job_script))
result = subprocess.run(submit_cmd, capture_output=True, text=True, check=True)
job_number = result.stdout.strip()
print(f"submitted job {job_number}")
