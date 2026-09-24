# Plan: RCP 8.5 projections + panel bootstrap (toward publication)

Agreed design from the 2026-08-12 session. Status: design locked, execution pending
(the Linux sandbox is down with `VM_DISK_SPACE_INSUFFICIENT`, so nothing has been run yet).

---

## 0. What already exists (and what is broken)

Both tasks are further along than expected, but two artefacts are unusable as-is.

**Bootstrap — largely done.**
`runs/Bootstrap/2026-06-16_21-25-59global_IC/` contains **500 completed country-cluster
replications** (`bootstrap_rep_status.csv`: reps 0-499, all `ok=True`, architecture `(2,)`,
5 initialisations each) plus `bootstrap_surfaces.npy`, `bootstrap_bands.npz`,
`bootstrap_draw_countries.npy`, `bootstrap_draw_indices.npy`. An earlier run
(`2026-06-16_12-10-08global_IC`) also exists.
The scheme is the correct one for this setting: resample **countries** with replacement,
preserve each country's full time series (so within-country serial dependence is untouched),
refit the within-transformed network on each resample.

Two problems:
- `paper/Chapters/115_Bootstrap.tex` states **200** replications; the run has **500**.
- The appendix *and* the sentence citing it in `5_Results.tex` are **commented out** of the
  submitted thesis. The compiled `s_final.pdf` has appendices A-O with no bootstrap appendix.
- The band-computation source is **missing** (only a stale `src/__pycache__/bootstrap.cpython-*.pyc`
  remains), so the code that produced `bootstrap_bands.npz` has to be rewritten.

**Projections — pipeline exists, inputs invalid.**
- `Benchmark/projections_Burke.py` is a faithful Python port of Burke's R replication
  ("Python analogue of `weightProj()`", "exactly like R", "In Burke's indexing"). It reads
  Burke's own inputs: `CountryTempChange_RCP85.csv`, `SSP_PopulationProjections.csv`,
  `SSP_GrowthProjections.csv` (OECD Env-Growth, SSP5), UN WPP population, and
  `bootstrap_noLag.csv` (Burke's 1,000 bootstrap coefficient draws). It carries an `n_boot`
  dimension through `GDPcapCC`/`GDPcapNoCC`, i.e. it already propagates uncertainty into the
  projections the way their Figure 5 does. **This is the machinery we should reuse.**
- `projections/MakeTemp.py` has a **fatal bug**: lines 120-121 hardcode `iso="AUS"` and
  `geometry = country_codes[iso]` *inside* the `for iso, geometry in country_codes.items()`
  loop, so every country is assigned Australia's geometry. The current
  `data/projections/NN/out/country_temp_change.csv` and `country_precip_change.csv` are
  therefore invalid (one value repeated).
- `projections/Projections.py` has further defects: `data` is used at line 67 before it is
  defined (it is local to `load_model`); `model` is never assigned at module level
  (`load_model` is defined but never called); `cfg.formulation` is attribute access on a dict
  returned by `yaml.safe_load` (should be `cfg["formulation"]`); and `temp_change`/`precip_change`
  are combined with `growthSSP5` **positionally**, with no merge on country code, so rows can
  silently misalign.

---

## 1. Decisions taken

| # | Decision | Choice |
|---|---|---|
| 1 | Bootstrap in the paper | **Include**, but diagnose and legitimately tighten first |
| 2 | Precipitation in projections | **Burke-exact main** (temperature only) **+ a precipitation variant** |
| 3 | Architecture inside bootstrap | **Hold `(2,)` fixed**, stated as conditional; fix real defects instead |
| 4 | Deliverable | **Two-panel figure** (global trajectory + regional decomposition) + headline table |

### Rationale worth preserving

**On (3).** Holding the architecture fixed is not a dodge if it is stated plainly: Burke's own
bootstrap is conditional on their quadratic functional form, so a band conditional on the
BIC-selected architecture is the apples-to-apples analogue. The comparison is then like-for-like
in its treatment of uncertainty as well as in its fixed effects. If a referee presses, the
fallback is re-selection among the top-3 architectures per replication (~3x cost, wider bands).

**On (2).** Burke's projection varies temperature only. Running the identical experiment makes
every difference in projected damages attributable to the response function alone. The
precipitation variant then isolates what the T x P interaction — the model's distinguishing
feature — contributes out of sample, without contaminating the headline comparison.

---

## 2. Execution plan

### Stage A — repair inputs
1. Fix `MakeTemp.py`: delete the two hardcoded `iso="AUS"` lines so the loop uses its own
   iterator. Regenerate `country_temp_change.csv` / `country_precip_change.csv`.
   Sanity check: the regenerated country temperature deltas must **not** be constant across
   countries, and should correlate strongly (rho > 0.9) with Burke's `CountryTempChange_RCP85.csv`
   where the two overlap. That correlation is also the validation that our CMIP6 SSP5-8.5
   pipeline reproduces their CMIP5 RCP 8.5 deltas.
2. Note the scenario nuance for the text: our NetCDF is
   `diff_tas_mon_mod_ssp585_192_ave_2081-2100_minus_1986-2005` — CMIP6 **SSP5-8.5**, the
   successor to CMIP5 **RCP 8.5**. The main figure uses Burke's own RCP 8.5 file, so this only
   affects the precipitation variant; state it explicitly in the appendix.

### Stage B — projections
3. Write `scripts/project_nn_rcp85.py` that reuses `Benchmark/projections_Burke.py`'s data
   preparation verbatim (SSP5 + OECD Env-Growth, UN WPP, `CountryTempChange_RCP85.csv`) and
   swaps only the response function:
   - growth impact for country i in year j = `f(T_i + delta_ij, P_i) - f(T_i, P_i)`
     (the delta form already used in `Projections.py`, which correctly cancels the fixed
     effects, trends and intercept since they do not depend on climate);
   - `f` = global NN surface (`model_visual`), then the regional NN surface, where each country
     is routed to its own region's output head. Region-specific time effects `nu_t(r)` cancel in
     the delta, so the regional projection is well defined.
   - **Merge on country code, never positionally** (this is the `Projections.py` bug).
   - Standardise `(T,P)` with the *training* mean/sd before the forward pass.
4. Run Burke's quadratic through the identical loop as the comparison arm, re-estimated on our
   sample, so the only difference is the response function.
5. Repeat holding temperature fixed and varying precipitation, and with both varying, for the
   variant panel.

### Stage C — bootstrap
6. Rewrite the band computation (source lost). For each of the 500 saved replication surfaces:
   - **re-center at the sample-mean temperature before taking percentiles.** The surface is
     identified only up to a level (the fixed effects absorb the intercept), so if the existing
     bands were computed on un-centered surfaces they contain pure, unidentified level shifts
     and are mechanically too wide. This is the first thing to check and the most likely single
     cause of excess width.
   - **filter non-converged replications.** `best_loss` ranges 0.0020-0.0047 across the 500 —
     more than a factor of two — which is too much heterogeneity to be sampling variation alone
     and suggests some replications landed in poor local optima. Diagnose the distribution,
     then trim on a pre-registered rule (e.g. drop reps with `best_loss` above the 95th
     percentile, or above a multiple of the median) and report the rule and how many were dropped.
   - Report both the raw and the cleaned band so the tightening is transparent, not cosmetic.
7. Propagate the (cleaned) bootstrap surfaces through the Stage B projection loop to obtain a
   95% band on projected damages — the direct analogue of Burke's `n_boot` treatment. This is
   what makes "our projection vs Burke's" a statistically meaningful comparison rather than two
   point estimates.

### Stage D — figure and text
8. Two-panel figure:
   - (a) global population-weighted % change in GDP per capita, 2024-2100, NN vs Burke, both
     with 95% bands;
   - (b) end-of-century damage by region from the regional surface vs Burke.
   Plus a table of 2100 headline numbers with confidence intervals (Burke's own headline is
   about -23% by 2100; ours goes next to it).
9. Reinstate `115_Bootstrap.tex`: uncomment, correct 200 -> 500, document the centering and
   convergence-filter rules, and add the projection band. Reframe the Results text: the honest
   claim is that the data do not pin down the functional form — the quadratic is not rejected,
   but neither is it required — while the Monte Carlo and the support-restricted bias table show
   the network is the safer default when the truth is unknown.

---

## 3. Open items for the next session

- Confirm which estimation runs are the paper's final global and regional models
  (`Projections.py` hardcodes `date_of_run = '2025-09-23'` for the regional model; the reported
  results are the global `(2,)` and regional `(4,)` runs — these need to be reconciled).
- Decide whether the regional bootstrap is needed (currently global only). If panel (b) carries
  bands, it is; that is a second 500-rep run.
- The sandbox must be back before any of this can be executed.
