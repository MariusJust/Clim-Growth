# Levels run on corrected data (2026-09-04), analysis and projections

## Item

Analyse `runs/estimation/2026-09-04_11-55-57_global_IC_country_trends_levels` (target_mode =
levels, log GDP per capita, Kalkuhl-Wenz setup) and produce the level-effect projections.

## Status

Complete. Run finished cleanly: 54 architectures, 10 initialisations each, 24h04m, 54 weight
files, `dynamic_model: False` as required.

**Headline: the levels specification puts the optimum at 11.67 C, close to Burke's 13.0 C and
6 C below our growth model's 17.64 C, and it yields economically modest impacts of roughly
plus or minus 3 percent rather than the growth model's plus or minus hundreds of percent.**

## Findings

### 1. Model selection

BIC selects `(2,)`, the same architecture the growth model selects, and decisively:

| rank | arch | BIC | dBIC |
|---|---|---|---|
| 1 | `(2,)` | -37832.63 | 0.00 |
| 2 | `(2, 2)` | -37789.12 | 43.52 |
| 3 | `(4,)` | -37785.86 | 46.78 |

The pre-correction June levels run selected `(2, 2)`; the climate-data fix moves levels
selection onto `(2,)`, aligning it with growth. BIC reproduced exactly from the saved weights
(recomputed minus recorded = 0.00).

Note AIC would select differently (`(4, 4)` at -42660.99); we use BIC throughout per D8.

### 2. Response surface, and why it matters

At median precipitation (1118 mm), centred at the sample mean (20.31 C):

| T (C) | levels: % effect on GDP pc | growth: pp/yr |
|---|---|---|
| 5 | +5.25 | -4.48 |
| 10 | +14.60 | -1.82 |
| 12 | +15.88 | -0.91 |
| 15 | +10.72 | +0.08 |
| 20 | +0.26 | +0.09 |
| 25 | +0.35 | -3.29 |
| 30 | +4.03 | -11.59 |

| specification | optimum |
|---|---|
| levels `(2,)` | **11.67 C** |
| growth `(2,)` | 17.64 C |
| Burke, our data | 14.55 C |
| Burke, published | 13.00 C |

The levels optimum sits 1.3 C below Burke's published value; the growth optimum sits 4.6 C
above it. This single parameter is what drove the whole NN-vs-Burke projection gap documented
in D16: the USA's baseline is 14.05 C, which is *below* the growth model's optimum (so warming
helps it) but *above* the levels optimum (so warming hurts it).

Incremental R2 over country FE, year FE and country quadratic trends: **0.00180** for levels
against 0.00298 for growth. The levels model explains even less of the within variation.

### 3. Projections, reported per D16

SSP5 / RCP 8.5, 162 countries, 2099, response pinned at 30 C, level effect with no compounding:

| statistic | levels | growth (for contrast) |
|---|---|---|
| **median country** | **+1.58%** | -67.66% |
| **world population worse off** | **40.4%** | 63.9% |
| countries worse off | 34.6% | 66.0% |
| Burke ratio of pop-weighted means | -2.90% | +14.25% |
| pop-weighted mean of country impacts | -1.22% | -24.92% |
| 25th / 75th percentile country | -2.62% / +2.42% | -87.76% / +63.03% |

Robustness (D16's leave-out-winners row) is where the two specifications differ most sharply:

| | all | drop top 1 | drop top 3 | drop top 5 |
|---|---|---|---|---|
| levels | -2.90% | -3.19% | -3.81% | -4.24% |
| growth | +14.25% | +4.97% | -10.37% | -23.52% |

The levels aggregate is **not** carried by a handful of countries. No single contributor exceeds
2.25 pp, and removing the top five moves the aggregate by 1.3 pp. The growth aggregate flips sign
on removing three. On the criterion D16 was written to enforce, the levels projection is the
better-behaved object by a wide margin.

Largest contributors: USA -10.0% impact (-2.25 pp), Japan -9.1% (-0.38 pp), France -7.9%
(-0.33 pp); Canada +7.6% (+0.21 pp), India +2.9% (+0.15 pp).

### 4. Why the magnitudes differ by two orders of magnitude

This is a specification choice, not an estimation result. In a growth specification a permanent
temperature change permanently shifts the growth *rate*, so the effect compounds over 89 years.
In a levels specification the same change shifts the *level* once. The two are the Burke and the
Kalkuhl-Wenz readings of the same data, and the paper can now quantify the gap rather than
assert it.

## What this does NOT do

- Does not produce confidence bands. There is no levels bootstrap run; bands would need a
  fresh job.
- Does not produce the mandated three-figure set: `plotly` and `kaleido` are absent from the
  sandbox. They are present in the user's venv, so `make_figures.make_all` runs there.
- Does not compare BIC across target modes. Growth and levels have different dependent
  variables and their information criteria are on different scales.
- Does not settle which specification the paper should lead with.

## Caveats to carry into the write-up

- The levels surface is **non-monotone above the sample mean**: it falls to +0.26 percent at
  20 C, then rises again to +4.03 percent at 30 C. That upturn sits in the region where the
  sample holds 3 observations out of 10,324, so it is functional form, not evidence. It does not
  affect the projections much (the cap at 30 C binds for 27 countries) but a referee will see it.
- The levels run used `output_initializer: he_normal` while the growth headline run used the
  code default `zeros`. The two targets are therefore optimised slightly differently. Worth a
  sentence, or a rerun of one of them for exact parity.

## Suggested follow-ups

1. Decide whether levels or growth leads the paper. The levels arm is far more robust under D16's
   own criterion, and its optimum agrees with the literature; the growth arm is the one that
   matches Burke's specification and invites the direct comparison.
2. If levels is to be reported with inference, dispatch a levels bootstrap.
3. Run the three mandated figures locally now that the venv is repaired.

## Artifacts

- Run: `runs/estimation/2026-09-04_11-55-57_global_IC_country_trends_levels`
- Script: `scripts/2026-09-08-levels-projections.py`
- Table: `paper/Tables/2026-09-08-levels-projections.csv`
- Compared against: `runs/estimation/2026-08-27_11-01-35_global_IC_country_trends`,
  `runs/estimation/2026-06-09_21-05-17_global_IC_country_trends_levels`
