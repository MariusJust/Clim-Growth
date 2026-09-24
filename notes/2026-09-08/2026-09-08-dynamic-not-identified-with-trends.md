# Country trends make the dynamic model unidentified

## Item

Why the dynamic model's estimated time variation is chaotic and
architecture-dependent, and whether removing country trends fixes it.

## Status

Resolved analytically. **Country-specific linear and quadratic trends annihilate any
smooth common function of t exactly, so the dynamic specification cannot identify a
smooth time-varying response while they are in the model.** Every dynamic run to date
(2026-07-02, 2026-09-01, 2026-09-03) carried them.

## Findings

### 1. The projection kills smooth time variation

Share of a smooth common time path surviving each nuisance projection:

| basis function | country FE only | + year FE | + country lin+quad trends |
|---|---|---|---|
| linear t | 90.794% | 0.000% | **0.000%** |
| quadratic t | 92.154% | 0.000% | **0.000%** |
| cubic t | 93.466% | 0.000% | **0.296%** |

A common trend lies in the span of the country-specific trends (sum the individual
ones), so the within projection removes it exactly. The network's t input can
therefore contribute nothing smooth; only high-frequency residual variation is left
for it to fit.

### 2. Identifying variation in temperature

Raw sd of temperature over the estimation sample is 7.2703 C.

| nuisance set | within rank | residual sd | % of raw variance left |
|---|---|---|---|
| country FE only | 196 | 0.5626 | 0.60% |
| country FE + year FE | 258 | 0.3964 | 0.30% |
| country FE + trends (current dynamic) | 588 | 0.3691 | 0.26% |
| country FE + trends + year FE (static) | 648 | 0.3450 | 0.23% |

Country FE alone remove 99.4 percent of the temperature variance. The trends remove
more than half of what remains, so dropping them multiplies the identifying variance
by about 2.3.

### 3. This explains the symptoms

Across architectures within about 220 BIC of each other in the 2026-09-03 run, with
the surface centred at the sample-mean temperature within each year:

| arch | BIC | params | time-varying share | corr(1961, 2023) | time-invariant amplitude |
|---|---|---|---|---|---|
| (8, 8, 4) | -53298.2 | 144 | 97.67% | -0.922 | 1.50pp |
| (8, 2, 2) | -53259.9 | 58 | 82.02% | +0.891 | 1.46pp |
| (8, 4, 4) | -53121.3 | 92 | 75.95% | +0.582 | 1.93pp |
| (2, 2, 2) | -53119.2 | 22 | 56.61% | +0.687 | 1.36pp |
| (4, 2, 2) | -53093.7 | 34 | 37.36% | -0.925 | 5.10pp |
| (2,) | -53076.9 | 10 | 54.91% | -1.000 | 2.18pp |

The time-varying share tracks parameter count, and the sign of the correlation
between the first and last year's response shape flips across architectures. Neither
is consistent with a stable time-varying relationship; both are consistent with
flexible functions fitting a non-smooth remainder.

## What this does NOT do

- Does not show the response is time-invariant. It shows the current specification
  cannot answer the question either way.
- Does not re-estimate anything. The conclusion is a property of the projection, not
  of any particular fit.
- Does not test the trend-free specification; that run (J2) is queued.

## Consequence

The dynamic model must be estimated with country FE only, matching eq (4.6) of
Bennedsen, Hillebrand and Jensen. J2 does this. The cost is a stronger identifying
assumption: without country trends, country-specific growth trajectories correlated
with climate load onto the climate response. That belongs in the text.

For a like-for-like dynamic-versus-static comparison, a static run on the same
country-FE-only basis would also be needed; it was dropped from the queue because the
question of interest is time variation, not fit.

## Artifacts

- Run analysed: `runs/estimation/2026-09-03_12-03-43_global_dynamic`
- Figure: `paper/Figures/Projections/dynamic_time_surface.pdf`
- Table: `paper/Tables/2026-09-08-dynamic-time-variation.csv`
- Script: `scripts/2026-09-08-dynamic-time-surface.py`
- Related: `notes/2026-09-04/2026-09-04-dynamic-he-normal-run.md`, PROJECT.md D12
