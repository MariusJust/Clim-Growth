# CLAUDE.md

Project conventions for *A Neural Network Approach to the Climate-Growth Relationship*.
Technical orientation (entry points, architecture, config keys) lives in `.claude/CLAUDE.md`;
this file holds the working rules.

## Paper weights: `results/paper/weights/` is the single source of truth

**Rule.** `results/paper/weights/` must always hold the weights that are actually in the current
version of the paper. The notebooks in `notebooks/` reproduce paper output from this folder, so if
it drifts from the run the text describes, every figure and number in the paper silently becomes
unreproducible.

**Whenever a new estimation run finishes, ask whether to replace the parameters in
`results/paper/weights/<formulation>/`.** Never copy them in silently, and never leave a new run
unmentioned. The question is always: *this run is newer than what is in `results/paper/weights/`,
should it become the paper's weights?*

When the answer is yes:

1. Move the existing contents to a dated archive folder. Never delete them.
2. Copy in `results.npy`, `parameters/`, and the run's `config/` or `.hydra/config.yaml`, so the
   provenance of the weights is recoverable later.
3. Record in `PROJECT.md` which run directory the weights came from, and on what date.
4. Flag every figure and table that now needs regenerating.

When the answer is no, say so in `PROJECT.md` too, with the reason. A run that was deliberately
not promoted is useful information.

### Current state, checked 2026-11-08: the folder is stale

```
results/paper/weights/global_model/results.npy     2026-02-24
results/paper/weights/regional_model/results.npy   2026-03-10
```

The runs the paper describes are from **2026-08-27**, so this folder is roughly six months behind.
Anything reproduced from it is not what the paper reports. See open task #44, "audit all analyses
that used `results/paper/weights/`".

## The locked global run

```
runs/estimation/2026-06-13_22-11-06_global_IC_country_trends
```

This is the run behind the paper's global results. It is locked: do not switch the global
figures to a newer run without saying so here and regenerating everything that depends on it.

## Name every exported figure after its run

An exported PDF should carry the run id, as
`Figures/Global/2026-06-13_22-11-06_global_IC_country_trends_CT.pdf` does. Provenance is then
checkable from the filename alone, with no code reading and no guessing from timestamps. This is
the cheapest reproducibility habit available, and the figures that lack it are exactly the ones
whose provenance took the longest to establish.

## Reproduction notebooks

`reproduction/` holds one notebook per figure, grouped as `global`, `regional`, `income`,
`monte_carlo`, `projections`. Each follows the same shape: install, paths and run, load, build,
plot, export. A new or changed paper figure should come with its notebook updated in the same
commit. See `reproduction/README.md` for the figure map and for which notebooks are still to be
ported out of `notebooks/scratch/`.

## Related reproducibility rules

- A figure in `paper/Figures/` must be newer than the run that produced it. Several are not: the
  global and regional surface PDFs are dated 2026-07-16, before the 2026-08-27 runs existed.
- `config/config.yaml` must reproduce the published results. It currently does not: it specifies
  `n_inits: 2` while the bootstrap behind the paper's bands used 3.
- Scripts that produce paper output belong in version control. Several of the CMIP6 and
  hidden-layer scripts are still untracked.
