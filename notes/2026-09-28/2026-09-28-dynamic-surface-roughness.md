# Why the dynamic surface is rough in 1980-2000, and how it differs from BHJ (2023)

Triggered by the top-6-average dynamic surface showing high-frequency structure
through 1980-2000 and a deep narrow trough at low temperature near 1990, against
the smooth Figure 17 of Bennedsen, Hillebrand and Jensen (2023).

Reference implementation:
`https://github.com/Sebastian-Jensen/A-Neural-Network-Approach-to-the-Environmental-Kuznets-Curve`
(`Functions/Dynamic_NN_model.py`, `Empirical Analysis/Estimate_models.py`).

## Item

Identify the cause of the roughness and the implementation differences against
the reference code.

## Status

Cause identified in the data, with high confidence. Four concrete code
differences documented. No model re-run yet.

## F1. The trough is the post-Soviet collapse, not a climate response

Country-demeaned growth (what the network must explain, given country FE),
averaged by temperature tercile. Cold is long-run mean below 17.3 C, warm above
25.7 C, 66 countries each.

| year | cold | mid | warm | cold - warm | n_cold |
|---|---|---|---|---|---|
| 1990 | -1.22 | -0.64 | -1.45 |  0.22 | 46 |
| 1991 | -6.31 | -2.74 | -0.90 | -5.40 | 62 |
| 1992 | -8.62 | -0.85 | -0.30 | -8.32 | 62 |
| 1993 | -4.46 | -0.98 | -1.45 | -3.02 | 62 |
| 1994 | -3.11 | -0.85 | -1.57 | -1.54 | 62 |

The standard deviation of `cold - warm` outside 1988-1996 is **1.31 pp**. So 1992
is a **6.4 sigma** deviation, 1991 is 4.1 sigma, 1993 is 2.3 sigma. The three
largest values in the whole 63-year sample are 1992, 1991 and 1993, with 2009
(the financial crisis, -2.74, also cold-skewed) fourth.

`n_cold` rising from 46 to 62 in 1991 shows the ex-Soviet states entering the
panel in the same year, so there is a compositional break on top of the shock.

**Mechanism.** The dynamic specification carries country fixed effects but no
time fixed effects and no country trends (`dynamic_include_time: false`,
`country_trends: False`). The climate net is therefore the only part of the model
that can absorb variation common to a year. A shock that hits cold countries and
spares warm ones is, to this model, indistinguishable from a temperature-by-time
interaction, and that is exactly what it fits. The 2009 crisis should produce a
smaller feature of the same kind.

This is a confound, not a finding. No climate mechanism operates on a three-year
window in 1991-1993 and reverses.

## F2. Four implementation differences against the reference code

### D1. Stopping rule. Ours is far more aggressive, and this is recent.

| | BHJ (2023) | Paper_1 now |
|---|---|---|
| learning rate | fixed 1e-3 | 1e-3 annealed to 1e-7 |
| `ReduceLROnPlateau` | **absent** | factor 0.3, patience 50 |
| `min_delta` | 1e-6 | 1e-7 |
| `patience` | 100 | 600 |
| typical epochs | short | 1364 to 10623 |

They run plain Adam at a fixed learning rate and stop after 100 epochs without a
1e-6 improvement. Early stopping is a well known implicit regulariser, roughly
shrinkage toward the initialisation, and it suppresses exactly the
high-frequency components that appear late in training.

**The D19 annealing fix is therefore a direct contributor to the roughness.** It
was still the right change (nesting violations proved the old fits were not at
optima, and D20 showed the architecture ranking was wrong), but it removed an
implicit smoothness prior that the reference implementation relies on. Full
convergence is what lets the network fit the 1991-1993 shock sharply.

### D2. Time enters raw, not standardised

In `Vectorize`:

```python
self.time_periods = tf.reshape(time_periods, (1, -1, 1)) - (time_periods[0] - 1)
time_vec = time_vec - self.time_periods_na_global
```

so time is an integer index 1, 2, ..., T fed directly to the first layer. Their
other input is `log(GDP/pop)`, of order 5 to 11. We use
`time_scaling: standardize`, giving t in about [-1.73, 1.73], with temperature
and precipitation also standardised to unit variance.

This changes the implicit smoothness prior in the time direction. Under
`he_normal`, a unit-scale first-layer weight produces a swish transition of width
about one unit of the input. On a raw 1..63 index that is roughly one year; on a
standardised index it is roughly 18 years. Which direction this pushes the fitted
optimum is not obvious a priori, because the network also starts far more
saturated on the raw scale. It is cheap to test and should be tested rather than
asserted.

Note also that on their scaling, time is numerically an order of magnitude larger
than income, which shrinks the fitted weight on the time direction relative to
ours, where all inputs have unit variance and time competes on equal terms with
temperature.

### D3. They plot one model; we plotted an average of six

`Estimate_models.py` selects a single architecture by BIC
(`where = np.where(BIC == np.min(BIC))[0][0]`) and Figure 17 shows that model.
The figure under discussion averages the top six architectures, which per D20
span dBIC 0 to 185.91, and which per the same decision sit in **different local
optima** (within-architecture BIC spread 229 to 482). Averaging six qualitatively
different surfaces produces local inconsistency that none of them has alone.

This is free to check and should be the first thing tried.

### D4. Different target, and it matters for the no-time-FE choice

Theirs is log CO2 per capita: a smooth, strongly trending level. Ours is the GDP
growth rate: noisy and shock dominated. Dropping time fixed effects is a mild
assumption for a smooth trending target and an aggressive one for a growth rate,
because all global business-cycle variation is then pushed into the climate net.
D12 recorded exactly this ("any future dynamic run must retain additive time
FE"); the 2026-09-26 run overrode it.

### Not differences

- Their BIC omits the fixed-effect parameter count while ours adds `rank(W)` to
  `m_effective`. A constant offset across architectures, so rankings are
  unaffected.
- They estimate country effects as N-1 explicit dummies; we use the FWL within
  projection. Equivalent in population.
- Activation (swish), optimiser (Adam), `he_normal` initialisation, `no_inits`
  selection by best BIC, and the 55-architecture grid all match.

## What this does NOT do

- No model re-run. Every claim about what a different specification would produce
  is a prediction.
- The D2 scaling argument is theoretical. It has not been tested.
- No check of whether the 2009 feature is actually present in the fitted surface.
- The reference paper's plotting script (`Production_region_plots.py`) is named in
  the README but returned empty from the repository, so the exact masking used in
  Figure 17 could not be read. The figure caption states that colouring
  distinguishes the region between the .05 and .95 income quantiles, so they do
  restrict the display to supported data per period; our figure applies no such
  mask.

## Suggested follow-ups, cheapest first

1. **Replot the single BIC-best architecture `(4,2,2)` instead of the top-6
   average.** Free, and matches the reference convention.
2. **Mask the surface to the supported temperature range per year**, following the
   Figure 17 caption. Free. Overlay the observed (T, year) scatter to expose how
   thin support is in the tails.
3. **Reinstate additive time fixed effects in the dynamic run** (D12). This is the
   substantive fix: it gives global shocks somewhere to go other than the climate
   surface. The cost is that the surface can then only be identified from
   deviations within a year, which is the trade-off D12 and D17 were circling.
4. **Test `time_scaling: none`** to match the reference implementation exactly.
5. **Robustness: drop the transition economies, or start the sample in 1995.** If
   the 1980-2000 structure disappears, F1 is confirmed directly.
6. Consider reporting the dynamic model with a shorter patience as a deliberate
   smoothness choice, but only with that stated openly; it is a regularisation
   decision, not a neutral default.

## Artifacts

- Data evidence computed from `data/MainData.xlsx` (GrowthWDI, TempPopWeight).
- Reference code: `Functions/Dynamic_NN_model.py`,
  `Empirical Analysis/Estimate_models.py` at the repository above.
- Related decisions: D12, D17, D19, D20.
