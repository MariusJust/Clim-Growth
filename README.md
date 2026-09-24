
## A Neural Network Approach to the Climate-Growth Relationship

This repository contains the full source code, data pipeline, and replication material for the paper.

The project implements a flexible neural-network–based panel data model to study how temperature and precipitation jointly affect economic growth. The framework relaxes restrictive parametric assumptions common in the climate–growth literature while retaining country and time fixed effects and a rigorous model-selection strategy.


## Repository layout

What each top-level folder is, and whether git protects it.

| folder | tracked? | what it is |
|---|---|---|
| `src/` | yes | The model. `Main.py` is the Hydra entry point; `models/` holds the global, regional and income formulations; `utils/` the shared helpers |
| `scripts/` | yes | Analyses and dispatchers. **See `scripts/README.md`**: the folder mixes libraries that must not move with dated one-offs that are finished |
| `config/` | yes | Hydra config. `config.yaml` is the single source of truth that `slurm.py` copies into each job directory |
| `paper/` | **separate repo** | The manuscript. Its own git remote, committed independently of this repository |
| `notes/` | yes | Dated working notes, one folder per day. The record of what was tried and why |
| `data/` | yes | `MainData.xlsx` is the estimation panel. `RawData/` the inputs, `archive/` superseded versions |
| `results/` | yes | Figures, tables and diagnostics that predate the `runs/` convention |
| `Benchmark/` | yes | `projections_Burke.py`, the ported Burke replication. Referenced by decision D5 by path, so it stays put |
| `notebooks/` | partly | Figure notebooks. `scratch/` is git-ignored |
| `runs/` | **NO** | Every estimation and bootstrap job, 18 to 71 hours of cluster time each. **Git-ignored: nothing here is recoverable.** See `runs/MANIFEST.md` for what each run is |
| `outputs/` | **NO** | Hydra's default directory for runs started locally rather than through `slurm.py`. Config snapshots and logs only, no results. Safe to clear |

`.gitignore` excludes `runs/`, `outputs/`, `results.npy`, `*.png`, `slurm.py`, `.venv/` and
`notebooks/scratch/`. Anything in those paths exists only on disk: back it up deliberately,
because a commit will not.

## Reproducing the environment (VS Code Devcontainer)

1. Install Docker desktop and VS Code
2. Install the “Dev Containers” extension in VS Code
3. Open this repository in VS Code and make sure you have a running engine in Docker
4. Press `F1` → **Dev Containers: Reopen in Container** 

Dependencies install automatically via `postCreateCommand`.

## Reproducing the environment (Docker)

1. Install docker on your machine 
2. run "docker build -t paper1 -f .devcontainer/Dockerfile ." from the command line 
3. run "docker run --rm -it -v "$(pwd)":/workspaces/Paper_1 -w /workspaces/Paper_1 paper1" from the command line. 

## Reproducing the results 

In the notebooks folder there exist 3 files for the global model, regional model and Monte Carlo simulation respectively. By running these notebooks, you can recreate the results shown in the paper.

**Weights: use `runs/estimation/`, not `results/`.** The weights under
`results/archive/2026-08-25-STALE-weights-pre-datafix/` predate the 2026-08-25 climate-data
correction and reproduce the wrong surface. The current headline weights are
`runs/estimation/2026-08-27_11-01-35_global_IC_country_trends/parameters/`; `runs/MANIFEST.md`
lists which run backs which result. The notebooks still point at the old path and will fail
until repointed, which is deliberate: a loud failure beats a quiet wrong figure.

Note also that `.devcontainer/` and `.dockerignore` are not currently present in the working
tree, so the Docker and Devcontainer instructions above cannot be followed as written. Recover
them with `git checkout HEAD~2 -- .devcontainer .dockerignore` if you want that route back.


## Contact

Marius Just
mjust@econ.au.dk
PhD Student, Econometrics
Aarhus University

For questions regarding the code, data construction, or replication, please open an issue or contact directly.


