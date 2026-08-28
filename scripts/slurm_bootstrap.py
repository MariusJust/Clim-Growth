"""Generate and submit the SLURM job for the country-cluster bootstrap.

Mirrors scripts/slurm.py, but points at src/bootstrap.py and writes into
runs/bootstrap/<job_id>/ instead of runs/estimation/.

    python scripts/slurm_bootstrap.py

Sizing note: on the corrected-data run the BIC-best node (2,) averaged
10.6 min per initialisation. With n_boot=1000 and n_inits=2 that is 2002 fits,
about 6.5 h wall on 55 cores. --time is set to 48 h for headroom.
"""
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf

ram_allocated = 200

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
RUNS_DIR = ROOT / "runs"
sys.path.insert(0, str(ROOT / "src"))

from utils.config import flatten_instance                      # noqa: E402

cfg = flatten_instance(OmegaConf.load(CONFIG_DIR / "config.yaml").instance)
boot = cfg.bootstrap

node_tag = str(boot.node).replace("(", "").replace(")", "").replace(",", "-").strip("-")
job_id = datetime.now().strftime("%Y%m%d_%H%M%S")
job_id += f"_boot{boot.n_boot}x{boot.n_inits}_{cfg.formulation}_node{node_tag}"
if cfg.country_trends:
    job_id += "_country_trends"

job_dir = RUNS_DIR / "bootstrap" / job_id
job_dir.mkdir(parents=True, exist_ok=True)
shutil.copytree(CONFIG_DIR, job_dir / "config")


def job_func(job_name, job_dir, root_dir):
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
    f.write(job_func(name, job_dir, ROOT))

print(f"n_boot={boot.n_boot}  n_inits={boot.n_inits}  node={boot.node}")
print(f"estimation run: {boot.run_dir}")
print(f"job dir: {job_dir}")

subprocess.run(
    ["sbatch", str(job_script)] if shutil.which("sbatch") else ["bash", str(job_script)],
    cwd=None if shutil.which("sbatch") else str(ROOT), check=True)
