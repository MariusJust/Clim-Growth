# Post-holiday analysis: three runs, two identification problems

## Item

Analyse the runs dispatched on 2026-09-08 and decide what they mean for the paper.

## Status

All three estimation jobs completed. The levels bootstrap never ran. Two of the three
findings are negative and both concern identification rather than fit.

## What landed

| run | spec | archs | wall | note |
|---|---|---|---|---|
| `2026-09-08_12-28-56_global_IC_country_trends_levels` | levels, zeros, trends | 54 | 26h12m | J1 |
| `2026-09-08_12-44-40_global_dynamic` | dynamic, he_normal, country FE only | 54 | 19h29m | J2 |
| `2026-09-08_13-40-58_global_dynamic` | identical to the above | 54 | 18h12m | duplicate |

The two dynamic runs are **bit-identical**: same seed, max absolute BIC difference
0.0000 across all 54 architectures. The pipeline is deterministic given the seed, which
is worth knowing but means the second run carries no information.

No bootstrap ran; the newest remains `2026-08-29_20-56-38global_IC`. The earlier levels
run `2026-09-04_11-55-57` has been deleted and is not in `runs/estimation/archive`, so
the he_normal levels surface survives only as the numbers recorded in
`notes/2026-09-08/2026-09-08-levels-run-analysis.md`.

## Finding 1: the levels response is not robust to the output initialiser

| levels run | BIC | optimum | response range | shape over supported range |
|---|---|---|---|---|
| he_normal (deleted) | -37832.63 | 11.67 C | 16.7% | interior peak near 12 C |
| zeros (2026-09-08) | **-37833.66** | **30.00 C** | 10.2% | monotone DECREASING 5 to 25 C |

The two are **1.03 BIC apart**, which is no basis for preferring either, yet they place
the growth-maximizing temperature 18 C apart. The zeros argmax sits at the grid edge,
driven by a +7.67% upturn at 30 C in the region holding 3 of 10,324 observations; within
the supported range that fit has no interior optimum at all. Round-trip verified:
recomputed BIC matches recorded to 0.00, incremental R2 0.00190.

BIC still selects `(2,)` decisively (runner-up +37.3), so architecture selection is
stable. It is the surface conditional on the architecture that is not.

**Consequence:** the levels passage in `5_Results.tex` claimed an 11.7 C optimum "close
to the 13 C reported by Burke". That claim holds under one initialiser and not the
other. The passage and the table's levels column are withdrawn behind a dated TODO.

## Finding 2: removing country trends did not identify the time variation

Predicted: the trends annihilate any smooth common function of t exactly (see
`notes/2026-09-08/2026-09-08-dynamic-not-identified-with-trends.md`), so removing them
should let the network express smooth time variation and produce a cleaner signal.
Observed: the spread across architectures got **wider**, not narrower.

| arch | BIC | params | time-varying share | corr(1961, 2023) | invariant amplitude |
|---|---|---|---|---|---|
| (2, 2, 2) | -55642.2 | 22 | 88.23% | -0.301 | 0.46pp |
| (4, 2, 2) | -55632.5 | 34 | **4.54%** | +0.986 | 7.83pp |
| (4, 4) | -55622.3 | 40 | 83.70% | -0.206 | 1.01pp |
| (4, 4, 4) | -55605.1 | 60 | 48.94% | +0.198 | 2.06pp |
| (4, 4, 2) | -55604.7 | 48 | **91.60%** | +0.810 | 0.43pp |
| (8, 2) | -55592.0 | 52 | 82.26% | -0.468 | 1.21pp |

`(4,2,2)` says the surface is 95 percent static with a large stable shape. `(2,2,2)`,
**9.6 BIC away**, says it is 88 percent time-varying with essentially no stable shape.
The previous run with trends spanned 37 to 98 percent; this one spans 4.5 to 92.

The trends were not the cause. Whatever is going on, the dynamic specification cannot
determine whether the climate response varies over time.

## Finding 3: the two problems may be the same problem

Findings 1 and 2 are both instances of near-equivalent fits implying qualitatively
different surfaces. That is the signature of a likelihood with many comparable local
optima, which would make it a property of the estimation problem rather than of either
specification. If so it also bears on the headline growth result, which has never been
checked this way.

## Action taken

- Levels passage and the table's levels column withdrawn in `5_Results.tex` behind a
  dated TODO recording both sets of numbers for restoration.
- `scripts/2026-09-24-multi-init-stability.py` written and functionally tested against a
  synthetic diagnostics directory. Reads `diagnostics/<node>/init_*.weights.h5`, reports
  the optimum and response range per initialisation over both the full grid and the
  5th-95th percentile support, conditional on BIC being within a tolerance of the cell's
  best, and overlays the curves.
- Levels bootstrap held. Bands around a point estimate that moves 18 C with the starting
  values would measure sampling variation while the larger source sits in the optimiser.
  The bootstrap also refits with `n_inits=2` and keeps the best, so each draw already
  mixes the two sources in an uncontrolled way.

## Next

Six-cell multi-init study, 50 initialisations each, `use_diagnostics: true`:
growth `(2,)`, levels `(2,)` and dynamic `(2,2,2)`, each under `zeros` and `he_normal`.
This separates within-initialiser spread from between-initialiser spread and, critically,
checks the growth headline for the first time. Then decide the canonical levels
specification and bootstrap that one.

## Artifacts

- Script: `scripts/2026-09-24-multi-init-stability.py`
- Figure (synthetic functional test): `paper/Figures/Diagnostics/2026-09-24-multi-init-stability.pdf`
- Table (synthetic functional test, marked as such in its header):
  `paper/Tables/2026-09-24-multi-init-stability.csv`
- Related: `notes/2026-09-08/2026-09-08-dynamic-not-identified-with-trends.md`,
  `notes/2026-09-08/2026-09-08-levels-run-analysis.md`
