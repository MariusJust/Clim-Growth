# Dynamic run with he_normal output initialiser (2026-09-03)

## Item

Analyse `runs/estimation/2026-09-03_12-03-43_global_dynamic`, the dynamic run with
`output_initializer: he_normal`, against the static BIC-best
`runs/estimation/2026-08-27_11-01-35_global_IC_country_trends` and the previous dynamic run
`runs/estimation/2026-09-01_14-04-01_global_dynamic`.

## Status

Complete. Run finished cleanly: 55 architectures, 10 initialisations each, 21h52m, all 55
weight files written.

**Headline: the dynamic model now wins on BIC by 32.26, reversing WS9. But the comparison is
not like-for-like, and the margin is the residue of two terms of order 700.**

## Findings

### 1. BIC ranking

| run | archs | BIC-best | BIC | AIC |
|---|---|---|---|---|
| static 2026-08-27 | 55/55 | `(2,)` | -53265.90 | -58016.80 |
| dynamic 2026-09-01 | 55/55 | `(4, 4, 4)` | -53201.48 | -57894.44 |
| dynamic 2026-09-03 (he_normal) | 55/55 | `(8, 8, 4)` | **-53298.16** | -58599.47 |

`he_normal` moved the dynamic model's best BIC by 96.68 in its favour. Verdict history across
three runs: static by 196, static by 64.4, now dynamic by 32.26.

Both headline BICs were reproduced exactly from the saved weights (recomputed minus recorded =
0.00 for each), so the numbers below are verified rather than back-solved.

### 2. The margin is a residue, and the comparison is not like-for-like

`multivariate_model.py:184` sets `include_time = not dynamic_model`, so the dynamic run carries
**no additive time fixed effects**. Its within-rank is 588 against the static model's 648.

| | static `(2,)` | dynamic `(8, 8, 4)` |
|---|---|---|
| within-rank | 648 | 588 |
| network parameters | 8 | 144 |
| m_eff | 656 | 732 |
| MSE | 3.193343e-03 | 2.973996e-03 |
| BIC | -53265.90 | -53298.16 |

Decomposition of the 32.26 gap:

- fit: `n * [log(MSE_static) - log(MSE_dynamic)]` = **+734.67**
- penalty: `(m_eff_s - m_eff_d) * log(n)` = **-702.41**

The dynamic model avoids 60 nuisance parameters, worth `60 * log(10324) = 554.53` BIC units of
penalty it never pays. The verdict is 4 percent of either term, so it is sensitive to how the
nuisance dimension is counted.

### 3. D12 pathology is gone

The 2026-07-02 run collapsed to 1-5 pp amplitude with a monotone increasing surface at every
date. This run does not:

| year | t index | amplitude | argmax T | shape |
|---|---|---|---|---|
| 1961 | 1 | 11.78pp | 30.00 C | interior peak |
| 1980 | 20 | 4.52pp | 25.10 C | interior peak |
| 2000 | 40 | 6.60pp | 11.20 C | interior peak |
| 2010 | 50 | 10.28pp | 0.30 C | interior peak |
| 2020 | 60 | 15.46pp | 9.60 C | interior peak |
| 2023 | 63 | 17.07pp | 9.60 C | interior peak |

Amplitude 4.52 to 17.07 pp against a static reference of about 12.2 pp, and an interior peak at
every date. The optimisation problem `he_normal` was meant to fix is fixed.

### 4. But the implied response is not interpretable, and t is not doing the year-FE job

The growth-maximising temperature wanders 30.0 -> 25.1 -> 11.2 -> 0.3 -> 9.6 -> 9.6 C across the
sample. That is not a time-varying climate response anyone can write down.

Testing whether the network's t input substitutes for the year effects it replaced:

| quantity | value |
|---|---|
| dynamic `f(Tbar, Pbar, t)` spread over sample | 16.10 pp |
| static year-FE spread | 14.75 pp |
| correlation between the two series | **-0.022** |
| sd ratio (dynamic time component / year FE) | 1.173 |

The time component is the same size as the year effects but **orthogonal** to them. The smooth
t-function is not reproducing the common year path; global year-to-year shocks are left in the
residual.

### 5. Capacity control

| model | BIC | MSE |
|---|---|---|
| static `(2,)` with year FE | -53265.90 | 3.193343e-03 |
| static `(8, 8, 4)` with year FE | -52283.77 | 3.131809e-03 |
| dynamic `(8, 8, 4)` without year FE | -53298.16 | 2.973996e-03 |

At matched capacity the dynamic model fits 5.31 percent better in MSE despite having no year FE.
So the gain is not purely extra parameters: the t input genuinely buys fit. But it buys it while
two things change at once (t added, year FE removed), so the source is not identified.

## What this does NOT do

- Does not establish that the dynamic model is preferred. The BIC comparison is between two
  different nuisance specifications, not between two models of the same object.
- Does not produce the mandated three-figure set. `plotly` and `kaleido` are absent from the
  sandbox, so `make_figures.make_all` cannot run here. It must be run on a machine with plotly.
- Does not test initialisation stability. Only the best of 10 inits per architecture is saved.
- Does not revisit D12, which remains correct as written: a dynamic run must retain additive
  time FE.

## Suggested follow-ups

1. **Rerun the dynamic model with `include_time=True`.** This is the like-for-like test and the
   one D12 asks for. With additive year FE retained, the within projection annihilates anything
   the network's t-function contributes at the common level, so t can only enter through its
   **interaction** with (T, P). That is exactly the time-varying climate response, cleanly
   separated from the common time path. Change is at `multivariate_model.py:184`; config-gate it
   rather than flipping it, so the current run stays reproducible.
2. Only if (1) still favours the dynamic model should the time-varying surface enter the paper.
3. If it does, report the argmax path with a caveat, or drop argmax and report amplitude only.

## Artifacts

- Run: `runs/estimation/2026-09-03_12-03-43_global_dynamic`
- Compared against: `runs/estimation/2026-08-27_11-01-35_global_IC_country_trends`,
  `runs/estimation/2026-09-01_14-04-01_global_dynamic`
- Relevant code: `src/models/global_model/multivariate_model.py:184` (`include_time`),
  `src/models/helper_functions/global_model/within_projection.py:84-89`
