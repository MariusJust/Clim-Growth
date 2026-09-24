# PROJECT — Paper 1: from thesis to publication

**Living document.** Stable filename (no date prefix) so future sessions can find it; update in
place rather than creating dated copies. Last updated 2026-08-25.

> **Resume protocol.** At the start of a session on this paper, read this file first, then the
> current work-stream document in `notes/`. Confirm in 1-2 sentences what was loaded before
> continuing.

---

## 1. Objective

Turn the master's thesis *A Neural Network Approach to the Climate-Growth Relationship*
(Marius Leisgaard Just, Aarhus University; supervisors Mikkel Bennedsen and Eric Hillebrand;
defended 14 August 2026) into a **publication-ready paper**.

The thesis is complete and defended. The publication gap is threefold: no out-of-sample
projections, no inference reported on the headline surface, and several known defects carried
over from the thesis draft.

**Target journal: not yet decided** — see Open Questions. This choice drives length, format,
and how much of the appendix survives.

---

## 2. What the paper is

Panel neural network mapping (temperature, precipitation) to growth in GDP per capita, on top of
country and time fixed effects and country-specific linear and quadratic time trends — the
*identical* panel structure to Burke et al. (2015), so the only difference from the canonical
specification is the shape of the climate response.

| Element | Detail |
|---|---|
| Estimation | Nonlinear least squares; linear block concentrated out via within transformation before training |
| Within transform | `P = I_n − Δ(Δ′Δ)⁺Δ′`, Δ = [Δ₁ country, Δ₂ time, Δ₃ = [diag(t)Δ₁, diag(t²)Δ₁]] ∈ ℝ^{n×(3N+T)}; Moore-Penrose because Δ is rank-deficient (Wansbeek-Kapteyn 1989; Penrose 1955/1956) |
| Recovery | `γ̂ = (Δ′Δ)⁺Δ′r`, minimum-norm least squares |
| Optimiser | ADAM, J = 10 random initialisations, keep lowest training loss |
| Selection | BIC over **55** architectures (rectangular + pyramid, Masters 1993) |
| Data | World Bank WDI GDP per capita 1960-2024, 182 countries, unbalanced, **N = 10,324**; CRU TS 0.5° climate, population-weighted (GPW v4, year-2000 weights) |
| Global result | `(2,)` selected; R² ≈ 0.233 vs quadratic 0.2323; FE + trends alone ≈ 0.230 |
| Regional | One shared representation, five region-specific output heads (Europe, Africa, Asia, Oceania, Americas); `(4,)` reported |
| Income | Median split of GDP per capita into low/high |
| Monte Carlo | 4 DGPs (linear, Burke, Leirvik/interactive, Trig/nonlinear), 100 reps each; architectures (2,), (4,), (4,2), (2,2) |

Thesis artefact: `s_final.pdf`, 45 pages, appendices A-O.
Length under the Aarhus convention: **66,655 signs** (57,055 running text + 12 floats × 800)
≈ 27.8 norm pages at 2,400.

---

## 3. Work streams

### WS1 — RCP 8.5 out-of-sample projections *(active)*
Project global and regional surfaces to 2100 under Burke's exact setup and plot against theirs.
Design locked. Detail: `notes/2026-08-12-projections-bootstrap-plan.md`.
**Extended 2026-08-25:** the headline object is now the **difference** (NN − Burke) with a CI on
the difference, across **SSP1-SSP5** (Burke's ported code already loops all scenarios), plus a
regional/income decomposition of who drives it. This is the supervisor's "quantify how much our
estimates change the downstream analysis", mirroring how Yuan et al. sold their results.

**Headline statistic locked 2026-09-04 (D16):** median country and population share harmed,
with Burke's ratio of population-weighted means alongside for comparability and never alone,
and the leave-out-winners row as the standing robustness check. Closes the "decide headline
projection specification given extreme fragility" question.

### WS2 — Panel bootstrap inference *(active)*
500 country-cluster replications already exist for the global `(2,)` model. Diagnose and
legitimately tighten the bands, then propagate them into WS1's projections. Same plan document.

### First result on corrected data *(2026-08-28)*
`runs/estimation/2026-08-27_11-01-35_global_IC_country_trends` (55 architectures, same
config, corrected climate). **The headline result survives the data fix.**
- BIC still selects **`(2,)`**; the top-six ordering is *identical* to the pre-fix run
  (BIC −53265.9 vs −53270.4).
- The response surface barely moves: the centred cross-section at median precipitation
  changes by at most **0.009** anywhere on 0-30 °C.
- Growth-maximising temperature **18.5 → 17.5 °C** (toward Burke's 13 °C, still above).
- Sample moments shift a lot (T 18.05 → **20.31 °C**, P 1094 → **1279 mm**), so the stable
  surface is a genuine robustness result, not an artefact of nothing having changed.
- The two neurons **swapped roles** (β = (−0.488, −0.025) → (−0.004, −0.639); index angle
  74° → 90.7°) while the surface stayed put — direct empirical confirmation of the
  permutation/near-cancellation identification caveat. Report surface-level objects, not
  raw γ/β.

### WS3 — Paper revisions *(pending)*
- Reinstate `paper/Chapters/115_Bootstrap.tex` (currently commented out); correct 200 → 500 reps.
- Decide whether the support-restricted MC table (`paper/Tables/mc_bias_rmse_support.tex`)
  replaces or supplements the full-grid table.
- Apply the outstanding fixes from `2026-07-28-proofread-report.md`.
- Reframe the contribution around inference (see Decision D1).

### WS5 — Finer-resolution (subnational) robustness *(active)*
Re-estimate the global model on the Earth Engine admin-1 panel (`data/ee_data.csv`,
semicolon-delimited, unit id `fid`, **1990 onward** — shorter T, much larger N than the
country panel) **with the within transformation**, then rewrite `113_EE_FinerData.tex`.

Blocked on a code change, not just a run:
- `src/models/global_model/model_architecture.py:63-64` (and `model_architecture_reg.py:105`)
  raise `NotImplementedError("within_projection (v1): wb only (no ee / group_trends_by_country)")`.
  The existing appendix results therefore came from the **explicit dummy-variable path**, which
  does support country-grouped trends; the within transformation has never been run on `ee`.
- `WithinProjector` stores dense `W` (n×p) *and* `B = (WᵀW)⁺Wᵀ` (p×n). At ee scale
  (n ≈ 90k+, p ≈ 3.3k with country-grouped trends) that is roughly 2.5 GB each. Fix: keep `W`
  sparse and never form `B` — apply `P v = v − W((WᵀW)⁺(Wᵀv))`. `WᵀW` stays small (p×p).
- `WithinProjector` builds trends as `Wc * t`, i.e. one per **unit**; it needs the `country_map`
  grouping so trends are per country, matching both the appendix spec and the main model.

Config for the run: `data_source: 'ee'`, `within_projection: true`,
`group_trends_by_country: true`, `model_selection: 'IC'`, and all 55 `nodes_list` entries
uncommented (only `(2,)` is currently active). Almost certainly a SLURM job.

### WS6 — Interpretability: what the network learns *(active)*
From the supervisor email 2026-08-25. Coefficients already extracted from the reported `(2,)`
global model. Headline findings: both neurons are temperature-dominated (9:1 and 18:1 in
standardised units) and their index directions differ by only **3.1°**, so the selected model is
effectively a **single-index (ridge) model**; one metre of annual precipitation offsets about
**1.16 °C** of warming along the index; and the curvature comes from **near-cancellation** of two
shifted swish ridges (`beta*gamma_T` = +0.239 vs −0.210). Individual `gamma`/`beta` are only weakly
identified (neuron permutation + near-cancellation), so lead with surface-level objects.
Remaining: multi-initialisation stability check, learned-factor time-series plots, write-up.
Detail: `notes/2026-08-25-supervisor-feedback-plan.md`.

### WS7 — Positioning against nonparametric alternatives *(active)*
Orimar's question: why not use off-the-shelf nonparametric methods? Answer built on WS6 — a
two-neuron network *is* a two-term projection-pursuit regression, so we are inside that
literature, with three properties standard packages lack (integrated panel-FE concentration,
scaling in input dimension, interactions without choosing a basis). Backed empirically by
estimating one tensor-product spline / GAM benchmark inside the within framework.

### WS8 — Dynamic model (time as network input) *(active)*
**Already run**: `runs/estimation/2026-07-02_12-01-08_global_dynamic`, all 55 architectures,
within projection, BIC. Dynamic best `(2,2)` = −53074.3 vs static `(2,)` = −53270.4, i.e. static
preferred by ~196 BIC points. **Gating check before reporting**: the dynamic path drops time fixed
effects (`include_time = not dynamic_model`), so `k_m` comparability across runs must be verified;
the comparison otherwise conflates "surface varies over time" with "are year effects needed".
Identification note: the dynamic surface is identified only through `t`×`(T,P)` interaction, since
common linear/quadratic components of `t` are absorbed by the country trends. Framing: a test for
**adaptation**, which neither Burke nor Karabiyik/Leirvik/Yuan attempt.

### WS4 — Submission prep *(not started)*
Journal choice, formatting, referee-proofing against the anticipated-questions list in
`2026-08-12-defence-qa-prep.md` — those 33 questions are, in substance, the referee report.

---

## 4. Decision log

Decisions are sticky: do not relitigate without a reason to.

| ID | Decision | Rationale |
|---|---|---|
| D1 | **Include** the bootstrap in the paper, but diagnose and tighten first | Absence of inference on the headline surface is the single most likely desk-reject point |
| D2 | Projections: **Burke-exact main** (temperature only, precipitation at baseline) **+ a precipitation variant** | Makes every difference vs Burke attributable to the response function alone; the variant then isolates what the T×P interaction adds |
| D3 | Hold `(2,)` **fixed** inside the bootstrap, stated as conditional | Burke's own bootstrap is conditional on their quadratic form, so this is the apples-to-apples analogue. Fallback if pressed: re-select among top-3 architectures per rep |
| D4 | Deliverable: **two-panel figure** (global trajectory + regional decomposition) + headline 2100 table with CIs | Standard publication format; covers both surfaces |
| D5 | Reuse `Benchmark/projections_Burke.py` and **Burke's own input files** rather than the in-house climate pipeline | It is a faithful port of their R replication; guarantees "exact same setup" and bypasses the broken `MakeTemp.py` |
| D6 | Support-restricted MC metrics use a **binary mask** (cell with ≥1 observation = weight 1, empty = 0) | Literal reading of "regions where we actually have data"; density weighting would conflate support with concentration |
| D7 | Compile LaTeX in a sandbox copy, copy the PDF back | The OneDrive mount is too slow for pdflatex and truncates `.aux` on abort, causing a corrupt-aux loop |
| D8 | EE robustness run uses **BIC and a single architecture**, matching the main text | One selection rule throughout the paper; the robustness check should differ from the headline model only in the data. Cost: the appendix's "AIC weights are less concentrated, so averaging is substantive" observation is dropped |
| D9 | The finer-resolution appendix is **must-have**, not backlog | It answers the "country aggregates mask within-country heterogeneity" critique the paper itself lists as a limitation. Half-fixing it is worse than cutting it, so it gets implemented properly |
| D10 | Projections cap the response at **30 °C** for every arm, `f(min(T,30))` | Burke's own choice ("so we are not projecting out of sample"); 30.1 °C is the warmest observed value; the saved bootstrap surfaces stop at 30 °C and linear extrapolation was measured to err by **1.92 pp/yr at 33.8 °C**. Measured cost of capping: only 0.3–0.7 pp at 2099. Supersedes the earlier "should not be capped" instruction |
| D11 | Report projections as **two panels** — population-weighted aggregate *and* median country / population harmed | With bands attached the 2099 aggregate spans −51% to +317% and cannot discriminate between arms; the aggregate also hides that the median country is ≈ −70% in every arm, including Burke's own published coefficients (−22.5% aggregate vs −76.8% median) |
| D13 | EE panel: **fixed effects per admin1 unit, trends per country** (linear + quadratic), global year FE | FE only absorb levels and admin1 units differ in level in ways correlated with climate, so per-unit FE is free and necessary. Trends per country because T=32: a per-unit quadratic trend would track that unit's own local warming and leave only weather. It also cuts the nuisance design from p=6893 to p=2733, making the within projection fit in memory. Matches what the 2026-06-01 EE run already used |
| D14 | EE observations are **weighted `w = 1/n_c`** so every country carries equal total weight | Unweighted, the estimate is weighted by how finely a country happens to be administratively subdivided — USA 179 units, Mali 1 — which is geography, not evidence. Cost is near-illusory: within-country correlation of the identifying variation (within-unit, net of year FE) is **0.82** (median 0.929), so a 179-unit country carries the information of ~1.22 independent units. Unweighted USA influence 7.83% -> 0.48%. Population weights are unavailable (the `population_density` column is genuinely density; country sums are 0.30-0.60 of actual population) |
| D15 | Weighted BIC uses **n = sum of weights = 6,624** (207 countries x 32 years) | It is what the weighted design asserts the sample to be, and is close to the true information content implied by rho=0.82. Using n=73,184 would treat 179 redundant US units as independent and under-penalise parameters by 1.91 each (~38 BIC units over a 20-parameter gap). Kish n_eff = 10,790 is the documented alternative |
| D12 | The 2026-07-02 dynamic run is **not** evidence on a time-varying response | It carries no additive time FE, so it fit the time trend instead of the climate response: temperature amplitude collapses to 1–5 pp (vs ~19 pp static) and the surface is monotonically increasing in T at every date. Any future dynamic run must retain additive time FE |
| D19 | **The learning rate never annealed: `EarlyStopping(patience=200)` fires before `ReduceLROnPlateau(patience=100)` can apply more than one cut, so `lr` never falls below 3e-4 and `lr_min: 1e-7` was unreachable dead config.** Fixed to `patience: 600`, `lr_reduce_patience: 50`. Selection is NOT contaminated, so the headline `(2,)` result stands; the defect is confined to large architectures that lose by thousands of BIC | Training is deterministic full-batch Adam (`y_train` has leading dimension 1, so `batch_size=1` is the whole panel and one epoch is one gradient step), so convergence depends entirely on annealing. Reaching `min_lr` from 1e-3 needs 8 cuts, hence 800+ stagnant epochs, against a 200-epoch leash. Two independent symptoms: within-MSE is non-monotone in width, which is impossible at a true optimum since a wider net can copy a narrower one and zero the rest (55 violated pairs in the 2026-08-27 headline run, worst +5.99%; 21 in the 2026-09-08 dynamic run, worst +3.43%); and steps-to-stop vary 11.5x across the ten initialisations of `(2,2,2)`. Scale: with n=10324, 1% of MSE is 103 BIC, so ~3% optimiser noise is ~350 BIC against a 9.65 BIC margin between the dynamic top two. Crucially the top six architectures in all three runs contain no violating architecture, and `(4,2,2)` does fit 0.98% better than `(2,2,2)` as nesting requires, so those two figures reflect non-identification of the dynamic surface (D17), not a failed optimisation: the 9.65 gap is a knife-edge cancellation of a -101.26 fit term against a +110.91 penalty. Also refactored `Multiprocess` to distribute over (node, init) rather than over nodes, since 6 architectures on 55 processes idled 49 cores and turned ~16h of work into an 80h job. Evidence: `notes/2026-09-24/2026-09-24-convergence-diagnosis.md` |
| D18 | Global model uses **`output_initializer: he_normal`**, not the code default `zeros` | The output layer has `use_bias=False`, so a zeros kernel gives `dL/dh = 0` exactly and the hidden layers receive no gradient on step one. In the 2026-09-03 pre-flight that made the dynamic fit early-stop after 31 epochs with a climate response of 0.1 pp, against 324 epochs and 25-57 pp under `he_normal`. The regional model has always used `he_normal`. Caveat: the 2026-08-27 growth headline run predates this key and used `zeros`, so growth and levels runs are not currently optimised identically, and D17 shows the choice can move the fitted surface by more than it moves BIC |
| D17 | **No claim about the shape of a fitted surface goes into the paper until it has survived a multi-init check.** Architecture selection by BIC is stable and can be reported as is; the surface conditional on that architecture cannot, until checked | Two levels runs differing only in the output-layer initialiser sit 1.03 BIC apart yet place the optimum 18 C apart (11.67 C versus 30.00 C, the latter pinned at the grid edge with no interior optimum over the supported range). Separately, the dynamic model's time-varying share ranges from 4.54 to 91.60 percent across architectures within 50 BIC, and the sign of corr(1961, 2023) flips. Both are near-equivalent fits implying qualitatively different surfaces, which points at many comparable local optima rather than at either specification. Evidence: `notes/2026-09-24/2026-09-24-post-holiday-run-analysis.md`. Supersedes the framing in D12: the binding constraint is not the absence of time FE but the identification of the surface itself |
| D16 | **Headline projection statistic = median country and population share harmed.** Burke's ratio of population-weighted means is reported alongside for comparability, never alone. The leave-out-winners row is the standing robustness check. Refines D11, which set the two-panel format but left the headline unnamed | Burke's aggregate is a ratio of means, not a mean of ratios, so it decomposes as `agg = sum_i w_i (GDPcc_i - GDPnocc_i) / sum_j w_j GDPnocc_j` and contributions scale with absolute dollars times population. Six rich cold countries carry the whole NN number: USA alone contributes +10.39 pp, Canada +6.84 pp, while India loses 95 percent of income across 1.15 billion people for only -4.92 pp. Dropping the top 3 contributors moves the NN aggregate from +14.25 percent to -10.37 percent; top 5 to -23.52 percent. Under every other aggregator the sign is firmly negative (pop-weighted mean of country impacts -24.92 percent, geometric mean -70.12 percent, pop-weighted median -75.76 percent, median country -67.66 percent, 63.9 percent of world population worse off). The pathology is Burke's, not ours: his own specification run through his own aggregator gives -7.22 percent headline against a -72.99 percent median country and 87.4 percent of population worse off. Evidence: `scripts/2026-09-04-aggregation-decomposition.py`, `paper/Tables/2026-09-04-aggregation-decomposition.csv` |

---

## 5. Known defects (open)

**Code**
- **`results/paper/weights/` IS STALE — do not use it.** The weights there do **not** reproduce the
  published figures. Figure 5 (`nn_vs_quadratic_percentiles`) is reproduced to within 0.005 by
  `runs/estimation/2026-06-13_22-11-06_global_IC_country_trends/parameters/(2,).weights.h5`
  (the run already referenced as `bootstrap.run_dir` and `nic.run_dir` in `config/config.yaml`),
  whereas `results/paper/weights/global_model/` gives a curve roughly twice as steep at the cold
  end and a *completely different* coefficient structure. The regional folder was previously found
  stale in the same way. **Canonical global run = `2026-06-13_22-11-06_global_IC_country_trends`.**
  Any analysis touching saved weights must be re-checked against this.
- `projections/MakeTemp.py` lines 120-121: `iso="AUS"` hardcoded *inside* the country loop, so
  every country receives Australia's geometry. `data/projections/NN/out/*.csv` are invalid.
- `projections/Projections.py`: `data` and `model` used before assignment; `cfg.formulation` is
  attribute access on a dict from `yaml.safe_load`; climate deltas aligned to SSP rows
  **positionally** instead of merged on country code.
- ✅ **RESOLVED 2026-08-30 — `src/bootstrap.py` restored by the user** from an external copy.
  The OneDrive → `C:\dev` move had silently dropped **four** never-committed source files:
  `src/bootstrap.py`, `src/network_information_criterion.py` (both restored),
  `src/utils/aic_weights.py` (still missing, but no longer imported anywhere — dead code), and
  `src/utils/config.py`, which was **reconstructed from bytecode** and is load-bearing
  (`bootstrap.py` imports `flatten_instance` from it). Root cause: git tracks only 64 files under
  `src/`, all from the stale capital-M `src/Models/` tree, so the entire current lowercase
  `src/models/` tree is untracked. **Commit it.**
- **The dynamic model has no additive time fixed effects** (`beta is None`), so the network
  competes with the time trend and wins: in `2026-07-02_12-01-08_global_dynamic` the
  temperature-carrying hidden unit saturates off (pre-activation −0.6 in 1961 → −25.2 by 2020),
  the response amplitude collapses to 1–5 pp against ~19 pp for the static model, and the surface
  becomes monotonically *increasing* in temperature. See `notes/2026-08-30-bootstrap-projections-bands.md`.
- The bootstrap saves **surfaces only, not weights**, so per-draw projections are limited to the
  saved (T, P) grid (T ≤ 30 °C, P ≤ 3000 mm). Fine under D10; would need re-running with weight
  saving if uncapped per-draw projections are ever required.
- Within projection is unavailable for `ee` data (`NotImplementedError`, see WS5), and
  `WithinProjector` materialises a dense `B` that will not scale to the subnational panel.

**Data**
- ✅ **FIXED 2026-08-27 — `data/MainData.xlsx` now holds corrected climate data.** Regenerated
  from raw grids by `scripts/2026-08-27-regenerate-climate.py` (pure-numpy NetCDF-3 reader in
  `scripts/netcdf3_min.py`; no netCDF4 needed). Original archived at
  `data/archive/2026-08-27-MainData-PREBUGFIX.xlsx`. Schema, sheet name, all 411 columns and all
  10,324 rows preserved — only `TempPopWeight` / `PrecipPopWeight` changed. Validation: unaffected
  countries reproduce the old values exactly (corr 1.00000, mean diff 0.005 C), agreement with
  Burke's UDel reference improves from corr 0.8955 / 1.765 C error to **0.9931 / 0.721 C**.
  Sample mean temperature 18.05 -> **20.31 C**. Refitting the Burke quadratic on the corrected
  data gives `b_T2 = -0.000473` against Burke's published `-0.000487` (was `-0.000551`).
  **All models still need re-estimating (task #47).** Historical description of the bug below.
- 🔴 ~~BLOCKING~~ (resolved): **population-weighting bug corrupted temperature AND precipitation for ~30% of the
  sample.** The grid-to-country aggregation drops invalid cells from the numerator but keeps their
  population weight in the denominator, so affected countries are scaled by
  `alpha_i = valid weight / total weight`. 58 of 181 checkable countries affected (3,137 of 10,324
  observations); mean alpha 0.762. Concentrated in tropical/coastal/island states — exactly the
  observations identifying the hot end of the response. Swapping to Burke's UDel series on an
  identical sample cuts the estimated curvature by 36% and reproduces Burke's published `b_T2`
  almost exactly. `ee_data.csv` does **not** have the bug. **Everything must be re-estimated after
  the fix.** Full diagnosis: `notes/2026-08-25-CRITICAL-climate-data-bug.md`.
- Seven of these have physically impossible values. Comparing our
  population-weighted CRU country means against Burke's UDel means: Solomon Islands 4.8 vs
  27.3 °C, Bahamas 9.6 vs 25.3, Brunei 11.0 vs 27.1, Honduras 11.4 vs 24.3, Vanuatu 11.6 vs 24.8,
  Belize 14.7 vs 25.8, Fiji 15.3 vs 23.7. All small-island or coastal-tropical, i.e. the
  signature of CRU's land-only 0.5° grid being mis-sampled by the grid-to-country mapping.
  387 of 10,324 observations. Excluding them barely moves the fitted quadratic (optimum
  14.32 → 14.45), so they are **not** driving results — but they are indefensible if a referee
  spot-checks, and the extraction should be fixed.
- Paper says "182 countries"; the estimation sample actually has **196** distinct `CountryCode`
  values over 1961-2023 (n = 10,324 is correct).

**Paper**
- `5_Results.tex`: Oceania "850 observations" contradicts N = 847 in all three tables.
- `3_Data.tex`: "10.324 observations" reads as a decimal; should be 10,324.
- `115_Bootstrap.tex`: says 200 replications, run has 500.
- `References.bib`: `CDD` entry author "New Mark" parses as first=New/surname=Mark (should be
  "New, Mark"); title has broken ligatures ("inﬂuenceon", "dailyprecipitation").
- Remaining consistency items in `2026-07-28-proofread-report.md` (year-range formats, US/UK
  spelling, adjectival hyphenation, `\qty{3} m`).

**Undocumented**
- How the Monte Carlo DGP coefficients were chosen is not stated anywhere in the paper. The
  values are hard-coded in `src/simulations/simulation_functions/Simulate_data.py::Surface()`.
  This needs a sentence in the text before submission.

---

## 6. Key file map

```
paper/                        NN.tex (main), Chapters/, Tables/, Figures/, References.bib
paper/slides/                 defence deck (26 counted frames; \eqscale + \eqtopsep knobs)
Benchmark/projections_Burke.py   faithful port of Burke's R replication  <- reuse this
projections/                  MakeTemp.py (BROKEN), Projections.py (buggy draft)
runs/Bootstrap/2026-06-16_21-25-59global_IC/   500 completed reps + surfaces + bands
results/paper/weights/        monte_carlo/{NN,Quadratic}/, regional/global model weights
scripts/2026-08-12-compute_mc_bias_support.py  support-restricted MC table generator
data/MainData.xlsx            TempPopWeight (°C), PrecipPopWeight (mm - divide by 1000 for m)
data/projections/old/         Burke's original inputs (RCP85, SSP, bootstrap_noLag.csv)
notes/                        work-stream plans
```

Deliverables produced this cycle: `2026-07-28-proofread-report.md`,
`2026-08-12-defence-qa-prep.md`, `2026-08-12-defence-flashcards.html`,
`notes/2026-08-12-projections-bootstrap-plan.md`.

---

## 7. Conventions

- **No em dashes or en dashes anywhere.** Hyphens are fine for ranges and compounds.
- Never delete; move to a clearly named archive folder. Show diffs and wait for confirmation
  before overwriting, renaming or restructuring.
- New files: `YYYY-MM-DD-descriptive-name` (living documents like this one are exempt).
- Evaluation grid: `linspace(0,30,90)` °C × `linspace(0.012,5.435,90)` m (`create_pred_input(mc=True)`).
- Surfaces are identified only up to a level; **centre at the sample-mean temperature** before
  comparing or taking percentiles.
- Sign count: Aarhus norm page — abstract + numbered sections + footnotes, equations as rendered
  text, each figure/table = 800 signs, references and appendix excluded (`count-signs` skill).
- Slides: two pdflatex passes for the frame counter to settle.

**Environment quirks**
- `src/` must be on `PYTHONPATH`; Hydra changes the working directory per run.
- The sandbox has no TensorFlow. TF-free helpers live in the `paper1-analysis` skill
  (`hdf5_min.py`, `surface_np.py`); anything needing TF must run on your machine or the cluster.
- OneDrive files may be cloud-only: `Read` hydrates them, bash may not see them.
- Bash cannot `rm`/`mv` on the mount; truncate with `: > file` instead.
- **Sandbox currently down** (`VM_DISK_SPACE_INSUFFICIENT`) — blocks all execution.

---

## 8. Open questions

1. **Target journal?** Drives length, structure and appendix survival. Candidates given the
   framing: *Journal of Applied Econometrics*, *Energy Economics* (where Bennedsen et al. 2023
   and Kahn et al. 2021 sit), *Journal of Environmental Economics and Management*.
2. Is a **regional bootstrap** needed? Currently global only. Required if panel (b) of the
   projection figure carries bands — that is a second 500-rep run.
3. Which estimation runs are **final** for global and regional? `Projections.py` hardcodes
   `date_of_run = '2025-09-23'`; the reported results are the global `(2,)` and regional `(4,)`
   runs. Reconcile before projecting.
4. Does the support-restricted MC table **replace** or **supplement** the full-grid table?
5. For the EE run: is the shorter sample (1990 onward, vs 1960-2024 in the main text) worth
   flagging as a caveat, and should the country panel be re-estimated on the matching window
   as a like-for-like comparison?
5. Any co-authors on the publication version, and does that change the framing?
