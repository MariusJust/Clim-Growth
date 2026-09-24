# Convergence diagnosis and the (node, init) refactor

2026-09-24. Triggered by two precipitation-percentile cross-section figures,
`(4,2,2)` and `(2,2,2)` from `2026-09-08_12-44-40_global_dynamic`, that are
9.65 BIC apart yet differ by roughly 5x in amplitude and disagree in shape.

## 1. Diagnosis: the learning rate never anneals

Training is deterministic full-batch Adam. `y_train` is reshaped to leading
dimension 1, so `batch_size=1` is the whole panel and each epoch is exactly one
gradient step. Convergence therefore depends entirely on the learning rate
schedule. The schedule cannot run:

```
EarlyStopping      monitor='loss'  min_delta=1e-7  patience=200
ReduceLROnPlateau  monitor='loss'  min_delta=1e-7  patience=100  factor=0.3  min_lr=1e-7
```

Going from `lr=1e-3` to `min_lr=1e-7` needs `log(1e-4)/log(0.3) = 7.65`, so 8
reductions, hence at least `8 x 100 = 800` stagnant epochs. EarlyStopping
terminates at 200. At most ONE cut ever fires, `lr` never falls below 3e-4, and
`lr_min: 1.0e-7` is unreachable dead config. Every fit stops inside a noise ball
whose radius is set by the learning rate, not at a stationary point.

## 2. Evidence

**Nesting violations.** A wider network can represent a narrower one of the same
depth exactly, by copying the units and zeroing the remainder. Within-MSE must
therefore be monotone in width at any true optimum. It is not:

| run | architectures | violated pairs | worst |
|---|---|---|---|
| `2026-08-27_11-01-35_global_IC_country_trends` (headline growth) | 55 | 55 | +5.99% |
| `2026-09-08_12-44-40_global_dynamic` | 54 | 21 | +3.43% |
| `2026-09-08_12-28-56_..._levels` | 54 | 6 | +0.69% |

Examples from the headline run:

```
(16, 8, 2) 0.0029557  <  (16, 16, 2) 0.00313286   (+5.99%)
 (8, 8, 2) 0.00296219 <    (8, 8, 4) 0.00313181   (+5.73%)
```

**Stopping time.** Wall time per initialisation is proportional to gradient steps
because the objective is deterministic and full batch. For `(2,2,2)` the ten
initialisations ran 227s to 2614s, an 11.5x spread. Median spread across all 54
architectures is 2.8x.

**Scale.** With `n = 10324`, a 1% change in MSE is `log(1.01) * n = 103` BIC
units. The demonstrated optimiser noise is up to 3.4%, roughly 350 BIC. The
margin used to select the dynamic winner is 9.65.

## 3. Scope limit: selection is NOT contaminated

The violations are confined to large architectures that lose on BIC by thousands
of units. `(32,32,32)` fits 18% better than `(2,)` in the headline run but
carries 2264 extra parameters, costing about 20,900 BIC. No convergence
improvement can bring those into contention.

The top six architectures in all three runs contain no violating architecture:

```
headline growth            dynamic
     (2,)   dBIC   0.0       (2,2,2)  dBIC  0.00
   (2, 2)   dBIC  50.2       (4,2,2)  dBIC  9.65
     (4,)   dBIC  67.4       (4, 4)   dBIC 19.86
(2, 2, 2)   dBIC 101.2       (4,4,4)  dBIC 37.06
```

So the paper's headline `(2,)` result is not at risk from this defect, and its
margin (50.2 BIC) is five times the dynamic run's.

**The two figures are not a convergence violation.** `(4,2,2)` correctly fits
0.98% better than `(2,2,2)`, as nesting requires. The 9.65 BIC gap decomposes as

```
fit term   dlog(MSE) * n = -101.26
penalty    dm * log(n)   = +110.91
total                    =   +9.65
```

That is a knife-edge cancellation, not a failed optimisation. The dynamic surface
appears to be non-identified, consistent with D17.

## 4. Code change: parallelise over (node, init)

`Multiprocess.parallel_execution` dispatched one task per architecture, with the
`no_inits` loop running serially inside it. (Note: `build_arg_list_ic` is dead
code on this path; `parallel_execution` ignores `self.arg_list`.) With 6
architectures on 55 processes that idles 49 cores and serialises every
initialisation, turning about 16 hours of work into an 80 hour job.

Changed:

- `src/models/global_model/run_experiment_ic.py`. Factored into `_fit_init` (one
  initialisation) and `_finalise` (selection plus artefact writing), shared by
  both execution paths so the selection logic cannot drift. Added `fit_one_init`
  and `reduce_from_rows` as the parallel entry points. Serial behaviour and the
  Monte Carlo branch are unchanged.
- `src/utils/parallel/multiprocess.py`. Added `parallel_execution_by_init`: pass
  one dispatches every (node, init) pair, pass two reduces to one winner per
  node. Gated on `parallel_over_inits`, IC selection, real data, global
  formulation, and `use_diagnostics: true`.
- `config/config.yaml`. `patience: 200 -> 600`, `lr_reduce_patience: 100 -> 50`,
  new `parallel_over_inits: true`.

`use_diagnostics: true` is REQUIRED for the parallel path: fitted models cannot
cross a process boundary, so the winner is reloaded from
`diagnostics/<node>/init_<j>.weights.h5`.

## 5. Regression, run BEFORE the pilot

The refactor is only sound if seeding is history independent: initialisation `j`
must give the same model in a fresh process as it does after `0..j-1` in a shared
one. `tf.random.set_seed(seed_value + j)` is called immediately before the model
is built, so it should be, but that needs evidence.

Run the same config twice, `parallel_over_inits: false` then `true`:

```yaml
optim:
  no_inits: 3
  use_diagnostics: true
  parallel_over_inits: false     # then true for the second run
model:
  dynamic_model: true
  country_trends: False
  quadratic_trends: False
  output_initializer: he_normal
  target_mode: growth
  nodes_list: ["(2,)"]
```

`python scripts/slurm.py` after each edit, then:

```
python scripts/2026-09-24-init-parallel-regression.py \
    --serial   runs/estimation/<serial_run> \
    --parallel runs/estimation/<parallel_run>
```

PASS requires max absolute per-initialisation BIC and AIC differences of exactly
0.0. Anything else means the parallel path is a different estimator and the pilot
must not be dispatched.

## 6. Pilot cells, dispatched in order

`scripts/slurm.py` chains submissions automatically via `.last_job_id`. Note that
`config.yaml` as committed has `country_trends: True`, which does NOT reproduce
the dynamic run.

Shared: `patience: 600`, `lr_reduce_patience: 50`, `use_diagnostics: true`,
`parallel_over_inits: true`.

**Cell A-dyn**, does the dynamic surface move across initialisations:

```yaml
optim:  {no_inits: 40}
model:  {dynamic_model: true, country_trends: False, quadratic_trends: False,
         output_initializer: he_normal, target_mode: growth}
nodes_list: ["(2,)", "(2,2,)", "(4,)", "(2,2,2,)", "(4,2,2,)", "(4,4,)"]
```

**Cell A-growth**, same question for the headline run as estimated:

```yaml
optim:  {no_inits: 40}
model:  {dynamic_model: false, country_trends: True, quadratic_trends: True,
         output_initializer: zeros, target_mode: growth}
nodes_list: ["(2,)", "(2,2,)", "(4,)", "(2,2,2,)", "(4,2,2,)", "(4,4,)"]
```

**Cell B**, does the fix remove the nesting violations (four nested pairs):

```yaml
optim:  {no_inits: 10}
model:  {dynamic_model: true, country_trends: False, quadratic_trends: False,
         output_initializer: he_normal, target_mode: growth}
nodes_list: ["(16,8,8,)", "(16,16,8,)", "(32,4,4,)", "(32,8,4,)",
             "(32,16,2,)", "(32,16,4,)", "(32,32,4,)", "(32,32,8,)"]
```

## 7. Pass criteria

- **Cell B**: nesting violations driven to zero. Direct test of the fix.
- **Diagnosis confirmation**: `lr_final` in `initialisations.csv`. Under the old
  schedule it should read 1e-3 or 3e-4 and never approach `lr_min`. Under the new
  one it should reach or approach 1e-7.
- **Cell A**: spread of the fitted surface across initialisations whose BIC lies
  within tolerance of that architecture's best. For the dynamic cells also report
  the time-varying share and `corr(1961, 2023)`.

Decision taken: `(2,2,2)` and `(4,2,2)` are reported side by side as the headline
exhibit rather than adjudicated. The within-architecture across-initialisation BIC
spread is computed anyway and held for the appendix, since it is the only measured
yardstick against which a 9.65 BIC gap can be judged.

Analysis lives in `notebooks/dynamic_model.ipynb`. The standalone
`scripts/2026-09-24-multi-init-stability.py` was deleted on 2026-09-24 at the
user's request; it is recoverable from commit `aa8573c`.
