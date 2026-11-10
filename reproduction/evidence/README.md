# Evidence

Plotly HTML exports that pin a paper figure to its run. A plotly HTML carries the full z grid,
so comparing it against a surface recomputed from `../weights/<run id>/` settles provenance
exactly, where a PDF cannot.

| File | Figure | Run | Status |
|---|---|---|---|
| `global_mean_CT_2026-06-13_22-11-06_global_IC_country_trends.html` | 4 | `2026-06-13_22-11-06` | confirmed by filename |
| `robustness_global_mean_CT_2026-06-13_22-11-06_global_IC_country_trends.html` | 4 | `2026-06-13_22-11-06` | confirmed by filename (2026-06-15 export) |
| `global_2026-06-13_22-11-06_global_IC_country_trends_CT.html` | 4 | `2026-06-13_22-11-06` | confirmed by filename |
| `2026-06-03_10-22-50_income_IC_country_trends_{High,Low}_CT.html` | 7 | `2026-06-03_10-22-50` | confirmed; `figure7.ipynb` asserts its mean surface matches to <1e-6 |
| `unverified/{Africa,Americas,Asia,Europe,Oceania}_mean_CT.html` | 6? | unknown | exported 2026-06-02/03, before the run the README names for Figure 6; not yet compared |

Copied from `results/images/` on 2026-11-10; the originals are still there.
