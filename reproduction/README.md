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
| 7 | `income` | `figure7.ipynb` | `archive/2026-06-03_10-22-50_income_IC_country_trends` | confirmed, to port |
| 8 | `projections` | `figure8.ipynb` | `2026-08-27_11-01-35_global_IC_country_trends` + bootstrap `2026-08-29_20-56-38_global_IC_bootstrap`, via `scripts/2026-09-28-projections-cmip6.py` | confirmed, ported. **Not the June run**, see below |
| 9 | `monte_carlo` | `figure9.ipynb` | `2026-05-15_10-52-58_MC_100reps` | to port |
| 10 | `monte_carlo` | `figure10.ipynb` | `2026-04-17_10-18-24_MC_100reps` | ported |

Every run in the table is frozen in `weights/<run id>/` (`results.npy`, `parameters/`, `config/`,
logs), and the notebooks read from there, not from the gitignored `runs/`. The Monte Carlo
benchmark surfaces for Figure 9 are in `weights/monte_carlo/`. Figure 7 also needs the
pre-bugfix data, frozen as `data/2026-08-27-MainData-PREBUGFIX.xlsx` and
`weights/2026-06-03_10-22-50_income_IC_country_trends/prebugfix_moments.npy`.
`evidence/` holds the plotly exports that pin a figure to its run.

The locked global run is `2026-06-13_22-11-06_global_IC_country_trends` (confirmed 2026-11-10).
Do not change it without updating `.claude/CLAUDE.md` and regenerating every figure that
depends on it.

**Figure 8 is on a different run, deliberately.** `scripts/2026-09-28-projections-cmip6.py`
hard-codes the August run `2026-08-27_11-01-35` (corrected climate data) and its bootstrap
`2026-08-29_20-56-38`. Decision 2026-11-10: keep it and flag it rather than regenerate the
projections from the June run. Figures 4 and 5 and Figure 8 therefore rest on different fits
and different data vintages.

## How the runs were identified

No figure except the locked global carries its run in the filename, so each was matched by
comparing the figure's creation date against the runs that existed at the time:

- **Figure 6**, created 2026-07-16. Two regional runs exist; only
  `2026-06-16_19-10-06` predates the figure. The other is from 2026-08-27. **Doubtful (checked
  2026-11-10):** the regional plotly exports, now in `evidence/unverified/`, were written
  2026-06-02/03, two weeks before `2026-06-16_19-10-06` finished, and the paper's regional PDFs
  differ from the PDFs exported next to them. Older regional runs exist in
  `runs/estimation/archive/` (newest before those exports: `2026-06-01_14-23-41`). Settle it by
  comparing each export's z grid against each candidate run, as for Figure 7.
- **Figure 10** is confirmed directly, and the timestamp inference was wrong. The run
  `2026-04-17_10-18-24_MC_100reps` keeps its own copies of `country_FE_MC.pdf` and
  `time_FE_MC.pdf`, and holds exactly the files the paper's plotting code asked for:
  `country_FE_table_Burke.csv`, `country_effect_relativeBurke.npy`, `time_FE_table_Burke.csv`,
  `time_trend_relativeBurke.npy`. The notebook reproduces both panels from it to the pixel.
- **Figure 9**, created 2026-05-21. Of the four 100-replication runs,
  `2026-05-15_10-52-58_MC_100reps` is the only one that predates it. Still inference.
- **Figure 7** is confirmed exactly, reproduced to float32 round-off. See the recipe below. The
  earlier timestamp guess, `2026-07-03_13-08-31`, was wrong: it is the five-quintile run, while
  the figure is the two-group High/Low split.
- **Figure 4** is confirmed directly: `2026-06-13_22-11-06_..._CT.pdf` names its run.

## Figure 7: the exact recipe

Reproduced to 5e-08 on the Low panel and 1.5e-07 on the High panel, which is float32
round-off in the stored export. No constant offset: the surfaces are plotted as computed.

- **Run** `runs/estimation/archive/2026-06-03_10-22-50_income_IC_country_trends`.
  Two groups, Low with 131 countries (reference IND) and High with 127 (reference USA),
  `country_trends` and `quadratic_trends` on, `within_projection` off.
- **Surface** the unweighted mean of the **top 10 architectures by BIC**, not the BIC-best
  single model:
  `(2,) (2,2) (4,) (2,2,2) (4,2) (8,) (4,2,2) (4,4) (8,2) (4,4,2)`.
  AIC gives the same ten, so this evidence does not separate the two criteria.
- **Standardisation** the **global** temperature and precipitation moments, not each group's
  own, taken from the pre-bugfix data:
  `mean_T 18.05276039, std_T 7.10028249, mean_P 1094.31669287, std_P 678.31324676`.
  These are exactly `data/prebugfix_moments.npy`.
- **Data** `data/archive/2026-08-27-MainData-PREBUGFIX.xlsx`, not the current
  `data/MainData.xlsx`. The current file was regenerated with the corrected population
  weighting on 2026-08-27, after this figure. Group membership happens to be unchanged
  (131 / 127), so the weights still line up, but the moments differ and the surface does not
  reproduce on the current data.
- **Grid** `T = linspace(0, 30, 90)`, `P = linspace(12.03731002, 3000, 90)`, precipitation
  capped at 3000 mm.

### How it was identified, after two wrong turns

The ground truth is `evidence/2026-06-03_10-22-50_income_IC_country_trends_{High,Low}_CT.html`
(originally `results/images/robustness/High_CT.html` and `Low_CT.html`), exported
2026-06-04. A plotly HTML carries the full 90x90 z grid as base64, so it pins the answer
exactly where a PDF cannot. The paper's `High_CT.pdf` and the renamed
`2026-07-16-high-income-CT-left-margin.pdf` are the same image recropped by a LaTeX pass, so
2026-07-16 is the crop date, not the run date.

Two mistakes cost most of the time, both worth avoiding next time:

1. **Span matching is too weak.** Comparing the vertical range of the surface produced
   plausible near-misses for several runs. Only the elementwise comparison, correlation plus
   the largest demeaned deviation, separates a true match from a coincidence.
2. **The per-group weight blocks are interleaved depth-major.** For two groups and depth `d`,
   hidden layer `l` of group `g` is `dense_{2l+g}`, not `dense_{g*d+l}`. Slicing them
   group-major silently produced wrong surfaces for every multi-layer architecture. After the
   hidden layers come one block per group of
   `[time FE, country FE, output head, linear trend, quadratic trend]`; the output head is the
   one with shape `(node[-1], 1)`.

Nothing in git history records the run. No `income.ipynb` ever existed, and no commit message
or tracked file mentions an income run directory, so the figure was made from a working copy
of `notebooks/regional_model.ipynb` that was never committed.

This is inference from timestamps, not proof, for every figure except 4 and 10. It is the best
available because the figures do not record their provenance, which is why `CLAUDE.md` now
requires the run id in the filename.

Figure 10 is the cautionary case. The newest run is not the right run, and the timestamp
argument picked the wrong one. What settled it was matching the *filenames* the plotting code
expected against what each run actually contains.

## Do not use the 2026-05-15 run for the time fixed effects

That run turned on country trends. The cross-country average of the per-country linear and
quadratic trends is itself a quadratic in `t`, which is collinear with the quadratic component
of the time effects, so the fit splits any quadratic in `t` between them arbitrarily. On that
run the estimated mean linear trend is 0.0103 against a true mean of -0.0001, with the
offsetting amount taken out of the time effects.

The consequence: comparing its raw time effects to the truth gives corr = **-0.75** and 0%
coverage, and looks like catastrophic failure. It is not. Adding the mean country trend back
recovers the identified object and gives corr = **0.999**. The 2026-04-17 run has no country
trends, which is why its time panel matches the truth in levels at corr = 1.000 with a 95%
band only 0.008 wide, exactly as the published figure shows.

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
