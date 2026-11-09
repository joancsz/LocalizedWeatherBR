# Results: degraded NE-US (PLAN_UPDATE_v2 §2.5)

Source: `notebooks/03_degraded_neus.ipynb` (Colab GPU). LaTeX: `$DATA_ROOT/tables/degraded_l{1,6,24,48}.tex` and `leads_neus.tex`.

Goal: separate the effect of a sparse network, station outages and a coarse grid from the effect of the tropical
climate, using the paper's NE-US data. Transformer + ERA5 Nearest, lead 1 h (6 h section below), 20 epochs, `use_is_real_mask: true` in
every run (so `base` is not the `neus_vit_20` run of `docs/repro_log.md`, which trained on filled values too).
These are reanalysis downscaling results, as are the paper's ERA5 / HRRR-A numbers.

| run | stations | outages | ERA5 grid | wind-vector error (m/s) | T RMSE (K) | Td RMSE (K) |
|---|---|---|---|---|---|---|
| base | 358 | none | 0.25° | 0.401 | 0.946 | 0.980 |
| sparse | 26 (spread, seed 0) | none | 0.25° | 0.338 | 0.871 | 0.864 |
| gaps | 358 | Brazil-like | 0.25° | 0.406 | 1.023 | 1.046 |
| coarse_100 | 358 | none | 1.0° | 0.398 | 0.943 | 0.943 |
| all | 26 | Brazil-like | 1.0° | 0.340 | 0.987 | 0.920 |
| sparse_gaps | 26 | Brazil-like | 0.25° | 0.342 | 0.978 | 0.949 |
| coarse_050 | 358 | none | 0.5° | 0.399 | 0.927 | 0.924 |

- Outage parameters: `target_real_fraction` 0.814 and `mean_gap_hours` 24 (from 8865 outages, median 3 h), measured in notebook 03 on the 26 Brazil
  stations (fraction of real `u` hours, mean outage length).
- Outages remove whole station-hours (all variables) and are drawn per year, test year included, so the gap runs are
  scored on fewer test hours (the notebook's table lists the real fraction per run).
- **Which row compares with Brazil:** `sparse_gaps`. Brazil's ERA5 is also 0.25°, so `all` (1.0°) is harsher than
  the Brazil setting; it bounds the grid effect rather than mimicking Brazil.

Each run is scored on its own stations and its own real hours (358 or 26 stations, real fraction 0.96 or 0.81), so the
rows are **not** directly comparable. The same runs scored on the 26 `sparse` stations and on the entries that are real
in every run (u/v 0.69, Td 0.65 of the test entries; 95% day-bootstrap CI):

| run | wind-vector error (m/s) | T RMSE (K) | Td RMSE (K) |
|---|---|---|---|
| base | 0.346 [0.334, 0.357] | 0.964 [0.945, 0.987] | 0.970 [0.944, 0.998] |
| sparse | 0.337 [0.325, 0.348] | 0.871 [0.852, 0.893] | 0.869 [0.843, 0.898] |
| gaps | 0.352 [0.340, 0.364] | 1.046 [1.024, 1.069] | 1.040 [1.012, 1.072] |
| coarse_050 | 0.345 [0.333, 0.356] | 0.959 [0.940, 0.981] | 0.924 [0.898, 0.953] |
| coarse_100 | 0.345 [0.333, 0.356] | 0.959 [0.940, 0.980] | 0.927 [0.903, 0.954] |
| sparse_gaps | 0.341 [0.329, 0.352] | 0.976 [0.954, 1.001] | 0.945 [0.917, 0.977] |
| all | 0.340 [0.328, 0.351] | 0.985 [0.963, 1.009] | 0.917 [0.888, 0.950] |

References for `base` (358 stations, `scripts/eval/baselines.py`, real only):

| forecast | wind-vector error (m/s) | T RMSE (K) | Td RMSE (K) |
|---|---|---|---|
| model (base) | 0.401 [0.389, 0.412] | 0.946 [0.928, 0.964] | 0.980 [0.956, 1.006] |
| ERA5 nearest | 2.595 [2.525, 2.661] | 1.764 [1.729, 1.801] | 1.915 [1.875, 1.957] |
| persistence | 0.403 [0.391, 0.416] | 1.150 [1.116, 1.187] | 0.842 [0.818, 0.870] |

Reading:
- At l = 1 h the model ties persistence on wind and loses to it on Td; it beats persistence only on T. Model biases
  are T -0.23 K, Td -0.27 K (persistence ~0).
- A coarser ERA5 grid (0.5° or 1.0°) does not hurt, and sometimes helps slightly: the model barely uses ERA5 at this lead.
- Thinning the network to 26 stations does not hurt the 26 kept stations (it helps, outside the CIs), so density is not
  the bottleneck here.
- Outages are the only factor that clearly degrades skill (T +0.08 K, Td +0.07 K vs base on the same entries).

## Lead 6 h and the lead sweep (to run)

At 1 h the model is persistence with a small temperature correction, so the grid factor cannot show. Notebook 03 adds
`base_l6`, `coarse_100_l6`, `sparse_gaps_l6`, `all_l6` (same factors at 6 h) and `base_l24`, `base_l48` (out to the
paper's 48 h), each scored with `scripts/eval/common_set.py` per lead and against ERA5 / persistence for `base*`.

_To fill: common-set table at 6 h, and model / ERA5 / persistence by lead (1, 6, 24, 48 h) for `base`._
