# Reproduction

One notebook per figure in the paper. Each has the same structure:

1. install the dependencies
2. set the paths and name the run
3. load the run and the data
4. build the quantity being plotted
5. plot it
6. write the PDF into `paper/Figures/`

The paper has **10 main figures**. Figures 1 and 2 are data maps, Figure 3 is a TikZ diagram in
the LaTeX source, so neither needs a notebook.

## Runs behind each figure

| Figure | Folder | Notebook | Run | Status |
|---|---|---|---|---|
| 4 | `global` | `figure4.ipynb` | `2026-06-13_22-11-06_global_IC_country_trends` | ported |
| 5 | `global` | `figure5.ipynb` | same locked global, plus the quadratic benchmark | to port |
| 6 | `regional` | `figure6.ipynb` | `2026-06-16_19-10-06_regional_IC_country_trends` | to port |
| 7 | `income` | `figure7.ipynb` | `archive/2026-07-03_13-08-31_income_IC_country_trends` | to port |
| 8 | `projections` | `figure8.ipynb` | locked global, via the CMIP6 script | ported |
| 9 | `monte_carlo` | `figure9.ipynb` | `runs/monte carlo/2026-05-15_10-52-58_MC_100reps` | to port |
| 10 | `monte_carlo` | `figure10.ipynb` | `runs/monte carlo/2026-05-15_10-52-58_MC_100reps` | to port |

The locked global run is
`runs/estimation/2026-06-13_22-11-06_global_IC_country_trends`. Do not change it without
updating `CLAUDE.md` and regenerating every figure that depends on it.

## How the runs were identified

No figure except the locked global carries its run in the filename, so each was matched by
comparing the figure's creation date against the runs that existed at the time:

- **Figure 6**, created 2026-07-16. Two regional runs exist; only
  `2026-06-16_19-10-06` predates the figure. The other is from 2026-08-27.
- **Figures 9 and 10**, created 2026-05-21 and 2026-07-07. Five Monte Carlo runs exist, four
  with 100 replications. `2026-05-15_10-52-58_MC_100reps` is the only 100-replication run that
  predates both.
- **Figure 7**, created 2026-07-16. Six income runs exist, all but one in
  `runs/estimation/archive/`. `2026-07-03_13-08-31` is the newest that predates the figure. Its
  `parameters/` holds per-quintile files (`(16, 16).Q1.Country_FE.csv` and so on), confirming
  the income formulation.
- **Figure 4** is confirmed directly: `2026-06-13_22-11-06_..._CT.pdf` names its run.

This is inference from timestamps, not proof, for every figure except 4. It is the best
available because the figures do not record their provenance, which is why `CLAUDE.md` now
requires the run id in the filename.

**Look in `runs/estimation/archive/` as well as `runs/estimation/`.** Most of the older runs
live there, including the one behind Figure 7, and it is easy to conclude a run is missing when
it is only archived.

## The paper's figures are not all on the corrected climate data

`data/MainData.xlsx` was regenerated with the corrected population weighting on **2026-08-27**.
Every run above predates that, so Figures 4 to 7, 9 and 10 are all fitted on the pre-correction
data. Newer runs exist for global, regional and income (all 2026-08-27), and the global BIC
ranking is unchanged between the two vintages, but the figures in the paper are the older ones.
This is a deliberate choice for now: the locked global run is the June one.

## Why most notebooks say "to port"

Only three figures have code in the repository that visibly produces them: Figure 4 from
`notebooks/global_model.ipynb`, Figure 8 from `scripts/2026-09-28-projections-cmip6.py`, and
Figure 10 from `notebooks/scratch/monte_carlo.ipynb`. The rest were made in `notebooks/scratch/`,
which is not organised by figure, so the exact cell has to be found by hand.

Unported notebooks raise `NotImplementedError` and name the source notebook to port from, rather
than silently producing something that is not the paper's figure.

## `_unused/`

Two notebooks for figures that turned out to be commented out in the LaTeX source: the
network-versus-Leirvik 3D comparison (`5_Results.tex:264`) and the 3×3 Monte Carlo grid. Kept in
case either is reinstated.
