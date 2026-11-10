# scripts/

Two different kinds of file live here, which is the main reason the folder is hard to
read. **Libraries** are imported by other scripts and must not be moved or renamed
without fixing their dependents. **Analyses** each answered one question on one day;
most are finished. Two of the libraries are disguised as dated analyses, which is the
single most confusing thing in this directory.

Import counts below are from 2026-09-24.

## Libraries: do not move or rename

| file | imported by | what it does |
|---|---|---|
| `hdf5_min.py` | **15 scripts** | TF-free reader for Keras `.weights.h5`. The keystone: nearly every analysis loads weights through it. Moving this breaks the folder |
| `_hdf5_pure.py` | `hdf5_min.py` | Pure-Python HDF5 primitives behind `hdf5_min` |
| `netcdf3_min.py` | `2026-08-27-regenerate-climate.py` | Minimal netCDF3 reader for the climate grids |
| `regional_surface.py` | 2 scripts | Per-region standardisation and TF-free forward pass for the regional model. Also carries `verify_mapping`, which checks the region-to-weight-block assignment against the data rather than trusting dict order |
| `2026-08-28-projections-corrected.py` | **5 scripts** | **MISNAMED.** This is the Burke projection harness (`ipolate`, `load_ours`, `within_fit`, `quad`), not a dated one-off. Later projection scripts `exec()` it by literal path, so renaming it means editing 5 dependents |
| `2026-08-30-quad-bootstrap.py` | `2026-08-30-projections-bands.py` | **Also misnamed.** Resumable cluster bootstrap of the Burke and Leirvik quadratics; supplies the cached coefficients in `data/bootstrap_quad_coefs.npz` |

## Dispatchers

| file | what |
|---|---|
| `slurm.py` | Estimation job dispatcher. **git-ignored**, so it exists only on disk. Keeps a personal sequential queue in `runs/estimation/.last_job_id` and chains each job after the previous one |
| `slurm_bootstrap.py` | Bootstrap dispatcher. Takes `--after <jobid>` to hold on another job. Hard-fails without `sbatch` rather than falling back to `bash`, which would run on the login node |
| `slurm_mc.py` | Monte Carlo dispatcher |
| `setup.py` | Package install for `src/` |

## Live analyses

Still produce something the paper or an open task depends on.

| file | produces |
|---|---|
| `2026-08-30-projections-bands.py` | `paper/Figures/Projections/projections_bands.pdf`, the projection figure in section 5 |
| `2026-09-04-aggregation-decomposition.py` | The D16 evidence: exact decomposition of Burke's aggregate, five aggregators, leave-out-winners |
| `2026-09-24-multi-init-stability.py` | Pending. Reads `diagnostics/<node>/init_*.weights.h5` and reports how far the optimum and response range move across initialisations. Blocks the levels write-up, see D17 |
| `2026-09-08-levels-projections.py` | Levels response surface and level-effect projections. Needs re-running against the current levels run |
| `2026-09-08-dynamic-time-surface.py` | Time-varying surface figure, temperature on x and years on y |
| `2026-09-04-regional-projections.py` | Per-region projections. Result was that the regional model is not usable for projection |
| `make_regional_bench_cross_sections.py` | Appendix M cross-sections. Fixed 2026-09-04 to use per-region standardisation |
| `2026-08-30-bootstrap-bands.py` | Response-function band figure from the bootstrap surfaces |
| `paper_surface.py` | Mirrors `create_pred_input`. **Currently imported by nothing**; keep or archive depending on whether you still generate surfaces this way |

## Spent one-offs

Each answered its question. Kept for provenance, safe to archive.

| file | what it settled |
|---|---|
| `2026-08-12-compute_mc_bias_support.py` | Monte Carlo bias over the support mask |
| `2026-08-25-learned-factors.py` | Plotted the learned factors A1, A2 against T and P |
| `2026-08-25-project-rcp85.py` | First RCP 8.5 projection attempt, superseded by the corrected harness |
| `2026-08-25-reproduce-burke.py` | Reproduced Burke's published coefficients, validating the pipeline |
| `2026-08-27-regenerate-climate.py` | Regenerated the climate data after the 2026-08-25 weighting bug |
| `2026-08-30-time-varying-surfaces.py` | Dynamic surfaces for the July run, using pre-bugfix moments |
| `2026-09-01-analyse-dynamic-run.py` | Analysis of the rescaled dynamic run |
| `2026-09-01-rolling-window.py` | 20-year rolling windows with cluster bootstrap CIs |
| `2026-09-01-make-briefing-pdf.py` | Built the supervisor briefing PDF |
| `2026-09-08-dispatch-holiday-jobs.py` | Batch dispatcher, superseded: jobs are submitted with `slurm.py` directly |
| `test_within_projector.py` | Verified the rewritten `WithinProjector` |
| `test_dynamic_preflight.py` | Pre-flight for the dynamic run before dispatch |
| `2026-08-28-analyse-c-drive.ps1` | One-off Windows disk audit |
| `2026-08-28-move-repos-off-onedrive.ps1` | One-off: moved the repos off OneDrive. Records the old `C:\dev\Clim_Growth` path |

## Conventions

- New files: `YYYY-MM-DD-descriptive-name.py`. Libraries get no date prefix, which is
  exactly the rule the two misnamed files above break.
- Scripts run from the repo root: `python scripts/<name>.py`.
- The sandbox has no TensorFlow. Anything needing TF runs on your machine or the
  cluster; TF-free weight loading goes through `hdf5_min.py`.
- New directories do not persist on the OneDrive-backed mount, so scripts that write
  figures should fall back to a directory that already exists.
