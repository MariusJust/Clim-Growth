# CRITICAL: population-weighting bug in the climate data (`MainData.xlsx`)

**Severity: blocking.** Temperature *and* precipitation are systematically understated for
roughly **30% of the estimation sample**. Every headline result in the paper is estimated on
this data. Found 2026-08-25 while investigating why our projections diverge from Burke (2015).

---

## 1. The bug

In the population-weighted aggregation from grid cells to countries,

```
T_it = sum_j ( T_jt * density_j ) / sum_j ( density_j )
```

the **numerator silently drops grid cells with missing climate values** (they enter a `nansum`
as zero) while the **denominator still counts their population weight**. The result is a
country-specific multiplicative dilution:

```
T_observed  =  alpha_i * T_true ,      alpha_i = (population weight on VALID cells) / (total population weight)
```

`projections/MakeTemp.py` lines 141-142 contain exactly this pattern:

```python
country_temp[iso] = np.nansum(tmp[mask] * find_closest(...)) / np.nansum(find_closest(...))
```

Countries whose land is poorly covered by CRU's land-only 0.5 degree grid — islands,
archipelagos, and heavily coastal states — lose a large share of their population weight to
invalid cells and are therefore pulled toward zero.

## 2. Evidence

**(a) The distortion is multiplicative and country-fixed.** Comparing our series against Burke's
UDel population-weighted temperature at the country-year level (6,641 overlapping observations,
161 countries, 1961-2010), the ratio `T_ours / T_UDel` has a **median within-country standard
deviation of 0.011**, and 99% of countries have sd < 0.05. The year-to-year *variation* is fine
(median within-country correlation 0.911); it is the **level** that is scaled.

**(b) The same alpha corrects precipitation.** Dividing precipitation by the alpha estimated
purely from temperature recovers published climatology across many independent countries:

| Country | alpha | P observed | P / alpha | Published |
|---|---|---|---|---|
| Solomon Islands | 0.176 | 575 | 3259 | ~3300 |
| Brunei | 0.406 | 1230 | 3030 | ~2900 |
| Samoa | 0.645 | 1851 | 2868 | ~2900 |
| Papua New Guinea | 0.724 | 1905 | 2629 | ~2800 |
| Philippines | 0.757 | 1751 | 2314 | ~2400 |
| Fiji | 0.642 | 1394 | 2173 | ~2100 |
| Belize | 0.566 | 1153 | 2035 | ~2000 |
| Bahamas | 0.377 | 501 | 1329 | ~1300 |

Nine independent countries, one shared scaling factor, agreement to within a few percent. This
is conclusive.

**(c) Physically impossible values.** Solomon Islands 4.8 C, Micronesia 7.2 C, Turks and Caicos
7.7 C, Grenada 8.1 C, Tonga 9.0 C, Bahamas 9.6 C, Brunei 11.1 C. All are tropical, sea-level.

**(d) The Earth Engine dataset does NOT have the bug.** `ee_data.csv` matches UDel almost exactly
for the same countries (Philippines 25.9 vs 25.9, Fiji 23.7 vs 23.7, Greece 15.0 vs 14.9,
Equatorial Guinea 24.5 vs 24.5). Only the very smallest islands remain imperfect there. So this
is specific to the CRU/World Bank pipeline behind `MainData.xlsx`.

## 3. Scale

Using UDel where available and Earth Engine otherwise as the reference (181 of 196 countries
checkable):

| Threshold | Countries | Observations affected |
|---|---|---|
| alpha < 0.97 (any dilution) | 58 | 3,137 of 10,324 (**30.4%**) |
| alpha < 0.90 | 37 | 1,905 (18.5%) |
| alpha < 0.85 | 29 | 1,487 (14.4%) |
| alpha < 0.70 (severe) | 14 | 677 (6.6%) |

Mean alpha among affected countries is **0.762**. Obs-weighted, temperature is understated by
about 4.4% overall — but the errors are **not random**: they are concentrated in tropical,
coastal, low-income countries, i.e. exactly the observations that identify the hot end of the
climate-growth response.

## 4. Why it matters for the results

On an **identical sample** (6,641 obs, 161 countries, 1961-2010, same growth data, same fixed
effects and trends), swapping only the temperature series in the Burke quadratic:

| Temperature source | b_T | b_T2 | optimum |
|---|---|---|---|
| CRU (ours, buggy) | +0.020533 | −0.000758 | 13.55 C |
| UDel (Burke's) | +0.013743 | −0.000484 | 14.20 C |
| Burke published | +0.012718 | −0.000487 | 13.06 C |

The buggy data produces **57% more curvature** than UDel. Note that UDel run through *our* growth
data reproduces Burke's published `b_T2` almost exactly (−0.000484 vs −0.000487), which confirms
the temperature series is the source of the discrepancy — not the sample period, not the
functional form, not the estimator.

This propagates directly into the projections, where our re-estimated quadratic gave +5.6% at
2099 against Burke's −22.8% on the identical harness.

## 5. What must happen

1. **Regenerate the country climate aggregation.** Fix the denominator to sum population density
   only over cells with valid climate data:
   `T_it = sum_{j: valid} (T_jt * d_j) / sum_{j: valid} d_j`.
   Also verify the grid-to-country mask and the population-grid alignment, since small islands may
   still have too few valid cells to be reliable at 0.5 degrees at all.
2. **Consider dropping or flagging micro-states** that have no adequate land coverage in CRU
   regardless of the fix (Solomon Islands, Micronesia, Turks and Caicos, Grenada, Tonga, ...).
3. **Re-estimate everything**: global, regional, income, model selection, the Monte Carlo
   calibration (its DGP coefficients come from this data), and the bootstrap.
4. **Re-check the interim correction option.** As a stop-gap for diagnostics only, dividing both
   T and P by the estimated alpha reproduces sensible values — but this is not a substitute for
   regenerating the data, because alpha is only estimable where a reference series exists.

## 6. Reproduce this diagnosis

Numbers above come from `data/MainData_est.csv` (slim export of `MainData.xlsx`),
`data/projections/old/in/mainDataset.csv` (Burke's UDel series) and `data/ee_data.csv`.
Country-level alphas are written to `/tmp/alpha_all.csv` by the diagnostic in this session.
