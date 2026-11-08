# Initialisation stability after the annealing fix

Runs analysed:

- `runs/estimation/2026-09-26_18-22-33_global_IC_country_trends` (static, Cell A-growth,
  `country_trends: True`, `output_initializer: zeros`, 40 inits)
- `runs/estimation/2026-09-26_18-23-41_global_dynamic` (dynamic, Cell A-dyn,
  `country_trends: False`, `output_initializer: he_normal`, 40 inits)

Both carry the D19 fix (`patience: 600`, `lr_reduce_patience: 50`) and
`use_diagnostics: true`, giving 6 architectures x 40 initialisations x 2 runs =
480 fits with per-initialisation records.

Cell B (the nesting check on large architectures) was not run.

## Item

Does the properly annealed optimiser change model selection, and is the fitted
surface stable across initialisations?

## Status

Answered, and the answer overturns the previous dynamic result.

## F1. The annealing fix works, everywhere

`lr_final = 1.0000000116860974e-07` for **all 480 initialisations** in all 12
cells. Under the old schedule the learning rate was pinned at 1e-3 or 3e-4 and
`lr_min` was unreachable. Epoch counts rose from roughly 200 to 2000 into the
600 to 10,623 range. D19 is confirmed and closed.

## F2. Dynamic model selection FLIPS

Best BIC over 40 initialisations, dynamic run:

| arch | best BIC | worst BIC | spread | dBIC |
|---|---|---|---|---|
| `(4,2,2)` | -55697.57 | -55357.26 | 340.31 | 0.00 |
| `(4,4)`   | -55674.00 | -55372.15 | 301.84 | 23.57 |
| `(2,2,2)` | -55650.85 | -55168.38 | 482.47 | 46.72 |
| `(2,2)`   | -55581.99 | -55232.99 | 349.00 | 115.58 |
| `(4,)`    | -55518.76 | -55289.17 | 229.59 | 178.81 |
| `(2,)`    | -55511.66 | -55270.28 | 241.38 | 185.91 |

Against the old run `2026-09-08_12-44-40` (10 inits, broken annealing), where
`(2,2,2)` led at -55642.16 with `(4,2,2)` 9.65 behind and `(4,4)` 19.86 behind:

**`(2,2,2)` and `(4,2,2)` swap places, and the margin grows from 9.65 in the
wrong direction to 46.72 in the right one.**

This matters directly for the two cross-section figures that started this work.
`(4,2,2)` was the smooth, concave, economically interpretable surface;
`(2,2,2)` was the low-amplitude one with the upward sloping 90th-percentile
panel. The properly converged optimiser selects the former. The earlier
contradiction was an artefact of stopping before convergence.

## F3. The flip survives the order-statistic objection

Best-of-40 is a more extreme order statistic than best-of-10, so the new run's
BIC is mechanically better regardless of the fix. Splitting the 40
initialisations into four independent blocks of 10 removes that advantage.
Minimum BIC per block:

| arch | inits 0-9 | 10-19 | 20-29 | 30-39 |
|---|---|---|---|---|
| `(4,2,2)` | -55682.24 | -55697.57 | -55696.78 | -55664.42 |
| `(4,4)`   | -55635.40 | -55674.00 | -55569.27 | -55614.46 |
| `(2,2,2)` | -55644.75 | -55562.88 | -55629.22 | -55650.85 |

`(4,2,2)` wins **4 blocks out of 4** against both rivals, by 37.5 / 134.7 / 67.6
/ 13.6 BIC over `(2,2,2)` and by 46.8 / 23.6 / 127.5 / 50.0 over `(4,4)`. The
flip is not an artefact of running more initialisations.

Note also the scale of best-of-10 sampling noise: 33 BIC for `(4,2,2)`, 88 for
`(2,2,2)`. Both dwarf the 9.65 BIC on which the old selection rested.

## F4. Static is identified, dynamic is not

Within-architecture spread across 40 initialisations:

| arch | static spread | dynamic spread | ratio |
|---|---|---|---|
| `(2,)`  | 28.31 | 241.38 | 8.5x |
| `(2,2)` | 41.22 | 349.00 | 8.5x |
| `(4,)`  | 15.07 | 229.59 | 15.2x |

The static objective behaves like a well-posed optimisation: every
initialisation lands in effectively the same place. The dynamic objective has
many well-separated optima. R2 makes the same point: static ranges 0.2302 to
0.2333 (a 1.3 percent relative band), dynamic ranges 0.054 to 0.112, a factor
of two.

This localises D17 precisely. The identification problem is a property of the
**dynamic specification**, not of neural-network estimation in this paper
generally.

## F5. The multimodality is discrete, not diffuse

Dynamic `(2,)` initialisations cluster at three distinct final losses:
0.0038435 (13 inits), 0.0039005 (19 inits), 0.0039343 (6 inits). These are
separated basins, not noise around one optimum. A larger initialisation budget
therefore buys basin coverage, not precision.

## F6. The paper's headline result is unaffected

Static run, top three by BIC:

| arch | new best BIC (40 inits) | new dBIC | old dBIC (10 inits) |
|---|---|---|---|
| `(2,)`  | -53264.03 | 0.00  | 0.00  |
| `(2,2)` | -53220.72 | 43.31 | 50.23 |
| `(4,)`  | -53198.55 | 65.48 | 67.39 |

The old headline run `2026-08-27_11-01-35` reached -53265.90 for `(2,)`. The new
best-of-40 is -53264.03, a difference of 1.87 BIC, comfortably inside that
cell's own 28.31 initialisation spread. **The selected architecture, the
ordering and the BIC level are all unchanged.** The response surface, the
projections and the bootstrap in the paper do not need revisiting on these
grounds.

## What this does NOT do

- **No surface comparison yet.** Everything above is BIC and R2 from
  `initialisations.csv`. Whether the `(4,2,2)` *surface* is stable across the
  initialisations that fit equally well is not established here; it needs the
  per-init weight snapshots. `scripts/2026-09-27-init-stability-analysis.py`
  does this and has not been run (the analysis sandbox was unavailable).
- **No nesting check.** Cell B was not run, so the 55 and 21 nesting violations
  recorded in D19 have not been retested under the fix. The three nested pairs
  inside these six architectures are a weak substitute and are computed by the
  same script.
- **Static `(2,2,2)`, `(4,2,2)`, `(4,4)` not tabulated.** The top three settle
  selection; the remaining three carry more parameters and were far behind in
  the old run. The script fills them in.
- **No claim that the dynamic model is now usable.** `(4,2,2)` leads `(4,4)` by
  23.57 BIC while within-architecture spread is 300 to 340. Selection among the
  top two is still not sharp, even though the block analysis favours `(4,2,2)`
  consistently.
- **Nothing written to the paper.** No `.tex` touched.

## Suggested follow-ups

1. Run `scripts/2026-09-27-init-stability-analysis.py` for the surface stability
   table and the three mandated figures.
2. Dispatch Cell B. It is now the only outstanding piece of D19, and the fix
   should be verifiable by nesting violations going to zero.
3. Reconsider the dynamic section of the paper around `(4,2,2)` rather than
   `(2,2,2)`. The time-varying share and `corr(1961, 2023)` in D17 were computed
   for the old, under-converged fits and should be recomputed.
4. Consider whether the levels run needs the same treatment. It was estimated
   under the broken schedule, and D17's levels contradiction (optimum 11.67 C
   versus 30.00 C for 1.03 BIC) may have the same cause as the dynamic flip.

## Artifacts

- Source: `runs/estimation/2026-09-26_18-22-33_global_IC_country_trends/diagnostics/*/initialisations.csv`
- Source: `runs/estimation/2026-09-26_18-23-41_global_dynamic/diagnostics/*/initialisations.csv`
- Pending: `scripts/2026-09-27-init-stability-analysis.py`
