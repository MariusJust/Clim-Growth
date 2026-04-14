import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
from omegaconf import OmegaConf
from hydra import compose, initialize_config_dir

# import numpy as np

# time_FE = np.load(f"../runs/monte carlo/2026-04-01_11-07-40_MC_1reps/time_FE.np.npy", allow_pickle=True)
# country_FE = np.load(f"../runs/monte carlo/2026-04-01_11-07-40_MC_1reps/country_FE.np.npy", allow_pickle=True)
# time_trend= np.load(f"../runs/monte carlo/2026-04-01_11-07-40_MC_1reps/time_trend.npy", allow_pickle=True)
# country_effect = np.load(f"../runs/monte carlo/2026-04-01_11-07-40_MC_1reps/country_effect.npy", allow_pickle=True)

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
RUNS_DIR = ROOT / "runs"

job_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
    cfg = compose(config_name="config_mc")

job_id += f"_MC_{cfg.mc.reps}reps"

job_dir = RUNS_DIR / "monte carlo" / job_id
job_dir.mkdir(parents=True, exist_ok=True)

shutil.copytree(CONFIG_DIR, job_dir / "config", dirs_exist_ok=True)

def job_func(job_name, job_dir, root_dir, ram_allocated, n_cpus):
    log_file = job_dir / "job.log"

    s = "#!/bin/bash\n"
    s += f"#SBATCH --job-name={job_name}\n"
    s += "#SBATCH --partition=q48\n"
    s += f"#SBATCH --mem={ram_allocated}G\n"
    s += "#SBATCH --nodes=1\n"
    s += "#SBATCH --time=200:00:00\n"
    s += "#SBATCH --ntasks-per-node=1\n"
    s += f"#SBATCH --cpus-per-task={n_cpus}\n"
    s += f"#SBATCH --output={job_dir / 'slurm-%j.out'}\n"
    s += f"#SBATCH --error={job_dir / 'slurm-%j.err'}\n"
    s += "\n"
    s += "set -euo pipefail\n"
    s += f'cd "{root_dir}"\n'
    s += "\n"
    s += "export TF_ENABLE_ONEDNN_OPTS=0\n"
    s += "export TF_CPP_MIN_LOG_LEVEL=3\n"
    s += 'export CUDA_VISIBLE_DEVICES=""\n'
    s += "\n"
    s += "export OMP_NUM_THREADS=1\n"
    s += "export OPENBLAS_NUM_THREADS=1\n"
    s += "export MKL_NUM_THREADS=1\n"
    s += "export VECLIB_MAXIMUM_THREADS=1\n"
    s += "export NUMEXPR_NUM_THREADS=1\n"
    s += "export TF_NUM_INTRAOP_THREADS=1\n"
    s += "export TF_NUM_INTEROP_THREADS=1\n"
    s += "\n"
    s += f'echo "===== JOB START =====" | tee -a "{log_file}"\n'
    s += f'date | tee -a "{log_file}"\n'
    s += f'hostname | tee -a "{log_file}"\n'
    s += f'echo "SLURM_JOB_ID=$SLURM_JOB_ID" | tee -a "{log_file}"\n'
    s += f'echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" | tee -a "{log_file}"\n'
    s += f'echo "===== PYTHON START =====" | tee -a "{log_file}"\n'
    s += (
        f'python -u src/simulations/monte_carlo.py '
        f'--config-path "{job_dir / "config"}" '
        f'--config-name config_mc '
        f'hydra.run.dir="{job_dir}" '
        f'2>&1 | tee -a "{log_file}" | '
        f'grep -Ev "All log messages before absl::InitializeLog|Unable to register cuDNN factory|Unable to register cuBLAS factory"\n'
    )
    s += 'status=${PIPESTATUS[0]}\n'
    s += f'echo "===== JOB END =====" | tee -a "{log_file}"\n'
    s += f'date | tee -a "{log_file}"\n'
    s += f'echo "Python exit status: $status" | tee -a "{log_file}"\n'
    s += "exit $status\n"

    return s

name = f"MC_{cfg.mc.reps}reps"
n_cpus = cfg.instance.n_process
ram_allocated = cfg.instance.ram

job_script = job_dir / "slurm.job"
with open(job_script, "w", encoding="utf-8") as f:
    f.write(job_func(name, job_dir, ROOT, ram_allocated, n_cpus))

use_slurm = shutil.which("sbatch") is not None # check if sbatch is available, if not run locally

if use_slurm:
    subprocess.run(["sbatch", str(job_script)], check=True)
else:
    subprocess.run(
        [
            "python",
            "-u",
            "src/simulations/monte_carlo.py",
            "--config-path",
            str(job_dir / "config"),
            "--config-name",
            "config_mc",
            f"hydra.run.dir={job_dir}",
        ],
        cwd=ROOT,
        check=True,
    )