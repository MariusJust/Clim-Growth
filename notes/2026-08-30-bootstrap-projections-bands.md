# 2026-08-30 — Bootstrap analysis, projections with bands, time-varying surfaces

## Current state

**Working on** — nothing in flight. All three deliverables of this session are complete.

**Completed**

1. **Bootstrap run analysed.** `runs/bootstrap/2026-08-29_20-56-38global_IC`:
   1000/1000 replicates succeeded, single node `(2,)`, `scheme=cluster`, `n_inits=3`,
   production optimiser settings (patience 200, min_delta 1e-07, 800–2000 epochs).
   The single-model change works. Point estimate is NOT in the file — reps run
   `range(n_boot)` with every replicate resampled, so the central line comes from
   the estimation run's saved `(2,).weights.h5`, which is Burke-consistent.

2. **Response function with bands** → `paper/Figures/Bootstrap/response_band.{pdf,png,csv}`
   Built by `scripts/2026-08-30-bootstrap-bands.py`.
   - point optimum **17.87 °C**, bootstrap median **17.53 °C** (well centred)
   - optimum 95% interval **[10.10, 27.98] °C** — wide
   - **96.5%** of draws have an interior optimum → concave shape is robust
   - response not distinguishable from flat at the extremes (at 30 °C: [−17.2, +0.5] pp)
   - band is zero at the centring cell (T=20.22, P=1287.8) *by construction*

3. **Projections with bands** → `paper/Figures/Projections/projections_bands.{pdf,png,csv}`
   Built by `scripts/2026-08-30-projections-bands.py` + `scripts/2026-08-30-quad-bootstrap.py`.

   | arm | 2099 point | 95% band | median country |
   |---|---|---|---|
   | Neural network | +13.2% | [−50.9, +316.6] | −66.7% |
   | Burke (our sample) | −7.8% | [−56.5, +165.6] | −72.1% |
   | Leirvik (our sample) | −9.0% | [−55.2, +170.7] | −68.7% |

   The aggregate is **uninformative at 2099** — a band spanning −51% to +317% cannot
   discriminate between arms. Panel B (median country) is where the signal is.

4. **Time-varying surfaces** → `paper/Figures/Surfaces/time_varying_surfaces.{pdf,png}`
   Built by `scripts/2026-08-30-time-varying-surfaces.py`. See the defect below —
   the figure is a diagnostic, not evidence about adaptation.

**Remaining**

- #48 headline projection specification (bands now make the case for the
  distribution panel rather than the aggregate)
- #29/#30 fold these figures into the paper; reinstate the bootstrap appendix
- #25 `MakeTemp.py` still broken; #34–36 EE within projection
- #50 commit the untracked `src/models/` tree

**Notes**

- Shell here caps at ~178 s and background processes do not survive between calls,
  hence the resumable chunked `2026-08-30-quad-bootstrap.py` (`--budget` seconds).

---

## Findings that change how results should be reported

### The dynamic model is degenerate as a climate model

`runs/estimation/2026-07-02_12-01-08_global_dynamic`, BIC-best `(2, 2)`. It has
**no additive time fixed effects** (`beta is None`), so the network spent its
capacity fitting the time trend instead of the climate response:

| year | t | level (pp) | range over T (pp) | unit-2 pre-activation |
|---|---|---|---|---|
| 1961 | 1 | −46.4 | 4.91 | −0.61 (alive) |
| 1980 | 20 | −51.8 | 1.10 | −8.52 (off) |
| 2000 | 40 | −84.4 | 2.10 | −16.84 (off) |
| 2010 | 50 | −115.1 | 3.13 | −21.00 (off) |
| 2020 | 60 | −153.3 | 3.49 | −25.16 (off) |

The temperature-carrying hidden unit saturates off; what survives passes through a
weight of −0.027. Response amplitude collapses to 1–5 pp against the static model's
~19 pp, and the surface is **monotonically increasing in temperature at every date**
(warming unambiguously good). This cannot support an adaptation claim and explains
BIC preferring static by 196. A usable version needs a dynamic run on corrected data
that *retains additive time FE*, so the time input and the trend stop competing.

### Off-grid extrapolation of the surface is unsafe

Bootstrap surfaces stop at T=30 °C and no weights were saved. Tested linear
extrapolation from the grid edge against the true `(2,)` network:

| T | true (pp) | linear extrap | error |
|---|---|---|---|
| 31 | 3.74 | 3.90 | 0.16 |
| 33.8 | −4.06 | −2.14 | **1.92** |
| 35 | −7.99 | −4.74 | 3.26 |

The swish net is still in its curved regime there, so extrapolation badly
understates damage. Everything is therefore capped at `f(min(T,30))` — which is
also Burke's own choice, and 30.1 °C is the warmest observed value in the sample.
Measured cost of capping vs not: only 0.3–0.7 pp at 2099.

### Time encoding in the dynamic model

Time enters **raw** as `year − (first_year − 1)` = 1…63 (panel 1961–2023); only
temperature and precipitation are standardised. `create_pred_input` builds
`arange(0, T+1)`, which starts at 0 and would shift the whole time axis by one
period — do not use it for this. 1960 maps to t=0, outside the trained range, so
the first panel is 1961.

The July run also needs the **pre-bugfix** standardisation moments
(T 18.053 ± 7.100, P 1094.3 ± 678.3), cached at `data/prebugfix_moments.npy`,
not the corrected ones (20.380 ± 7.231, 1281.6 ± 833.7).

---

## Files

**Created**
- `scripts/2026-08-30-bootstrap-bands.py`
- `scripts/2026-08-30-quad-bootstrap.py` (resumable; optimised fit verified to
  reproduce the reference `within_fit` to 8.8e-17, ~3× faster)
- `scripts/2026-08-30-projections-bands.py`
- `scripts/2026-08-30-time-varying-surfaces.py`
- `data/bootstrap_quad_coefs.npz` (1000 Burke + Leirvik cluster draws)
- `data/prebugfix_moments.npy`
- figures listed above

**Modified**
- `PROJECT.md` — decision log D10–D12, defect updates

## Verify manually

- Projections use **162** countries, not the earlier 165: baseline temperatures are
  inner-joined on our own data rather than back-filling three missing countries with
  Burke's UDel values. Point estimates shift slightly (Burke −8.93% → −7.84%).
- `src/utils/config.py` is still a **bytecode reconstruction**, and it is load-bearing
  (`bootstrap.py` imports `flatten_instance` from it). Verified equivalent, but it is
  not the original file.
