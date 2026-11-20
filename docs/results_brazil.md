# Results: NE Brazil (P4, E2–E4)

Source: `notebooks/04_brazil_experiments.ipynb` (Colab GPU). LaTeX versions: `$DATA_ROOT/tables/{e2,e3,e4_density,e4_filter,leads_brazil}.tex`
from `scripts/eval/make_tables.py`.

**Read every number below as reanalysis downscaling.** At every lead ERA5 is taken at the target hour, so the model
corrects ERA5 towards the stations at a time when ERA5 already exists; that matches the paper's ERA5 / HRRR-A results,
not a forecast (only the paper's HRRR-F is one).

Setup: 26 INMET stations (control ratio 0.6, `docs/repro_log.md` P3), train 2019–21, validation 2022, test 2023,
`use_is_real_mask: true`, 50 epochs. **Lead 6 h** for E2–E4 (decided 2026-11-09):
at 1 h the NE-US runs tie persistence and ignore the ERA5 grid (`docs/results_degraded_neus.md`). Metrics on real observations only; brackets are 95% intervals from a block
bootstrap over days.

## E2: model vs ERA5 interpolation and persistence

Lead 6 h, 26 stations, test year 2023 (`ne_brazil_{vit,gnn}/baselines.csv`, table `e2.tex`).

| forecast | wind-vector error (m/s) | T RMSE (K) | Td RMSE (K) | T bias (K) | Td bias (K) |
|---|---|---|---|---|---|
| Transformer + ERA5 Nearest | **1.057** [1.043, 1.070] | **0.991** [0.964, 1.016] | **1.224** [1.200, 1.247] | 0.08 | 0.13 |
| GNN (Delaunay) + ERA5 grid | 1.154 [1.138, 1.169] | 1.393 [1.332, 1.465] | 1.528 [1.487, 1.573] | -0.14 | -0.40 |
| ERA5 nearest | 2.541 [2.502, 2.580] | 1.328 [1.304, 1.351] | 1.818 [1.788, 1.846] | 0.16 | 0.62 |
| persistence | 1.860 [1.831, 1.890] | 4.841 [4.763, 4.919] | 2.567 [2.494, 2.650] | -0.02 | 0.03 |

- The Transformer beats both references on every variable: wind error is 58% below ERA5 and 43% below persistence.
- The GNN beats both references on wind and Td but is **worse than ERA5 on T** (1.39 vs 1.33 K).
- At l = 1 h persistence is a near-ceiling on NE-US (0.403 m/s wind vs the model's 0.401). In Brazil the model beats it
  at 1 h too (0.870 vs 1.002 m/s, lead sweep below).

### Why the GNN loses on temperature

Checked on the test pickles (same samples, targets and real mask as the Transformer):

- **Undertrained.** Validation loss was still falling at epoch 50 (0.973 at 40, 0.957 at 50) with a wider train/val gap
  than the Transformer (0.844 / 0.957 vs 0.815 / 0.827). The NE-US reproduction showed the same (`docs/repro_log.md`:
  GNN "further from convergence than the Transformer").
- **Broad, not one bad station.** The GNN's T RMSE is above the Transformer's at all 26 stations and at every local
  hour (1.24–1.56 K vs 0.71–1.25 K). Dropping the worst station (26) only moves 1.39 to 1.34 K.
- **Shared outlier tail.** 1724 real T entries have |error| > 5 K, 483 of them at station 26. The Transformer misses
  the same entries by about the same amount, so these look like bad observations rather than a GNN failure. Station
  26 has T RMSE 2.25 K (Transformer) and 2.81 K (ERA5) and is worth an INMET quality check. Without those entries the
  GNN T RMSE is 1.16 K.
- **Worst months: December (2.11 K) and March (1.73 K).** 4% of the GNN's squared T error falls on 2023-12-23 alone.

Reading: the GNN gap is mostly training budget, not a property of the Brazil data. A longer GNN run (100 epochs) would
settle it. Until then, report the Transformer as the main model.

## E3: sea breeze (main result)

From `ne_brazil_vit/diurnal/` (`wind_error.png`, `uv_cycle.png`, `stations.csv`, table `e3.tex`), lead 6 h, local time UTC−3.

- Coastal = within 20 km of the Natural Earth 10 m coastline (decided 2026-11-08): **3 coastal, 23 inland**. The
  counts at 10 / 30 / 50 km are 2 / 4 / 4.
- Reading the figure: ERA5 error is discontinuous at 09/10 and 21/22 UTC (06/07 and 18/19 local), the boundaries of
  ERA5's 12 h assimilation windows; seen on NE-US too, not a pipeline artefact.

Mean wind-vector error over the local day (m/s):

| stations | model | ERA5 | persistence | model peak hour (local) |
|---|---|---|---|---|
| coastal, < 20 km (3) | **0.668** | 3.255 | 1.511 | 11 |
| inland (23) | **1.106** | 2.443 | 1.907 | 15 |
| coastal, < 30 km (4) | **0.713** | 3.107 | 1.666 | – |
| inland at 30 km (22) | **1.110** | 2.453 | 1.891 | – |

**Robustness with 3 coastal stations.** Per station (`stations.csv`), the model beats ERA5 and persistence at **each**
coastal station:

| station | coast (km) | model | ERA5 | persistence |
|---|---|---|---|---|
| 9 | 1.8 | 0.44 | 3.94 | 0.78 |
| 1 | 4.1 | 0.93 | 2.49 | 1.99 |
| 2 | 10.9 | 0.71 | 3.13 | 1.88 |
| 17 (only added at 30 km) | 27.3 | 1.02 | 2.05 | 2.78 |

Network-wide, the model beats ERA5 at 25 of 26 stations and persistence at 25 of 26. Station 14 is the exception for
both (model 2.48, ERA5 1.97, persistence 3.06 m/s). Widening "coastal" to 30 km changes nothing qualitatively.

**What the model corrects (`uv_cycle.png`, coastal mean u and v by local hour).** The model reproduces the observed
mean cycle almost exactly. ERA5's error is mostly **magnitude, not timing**:

- u (onshore easterly): ERA5 has the right phase (diurnal anomaly correlation 0.98 with observed, minimum at 14 vs 13
  local) but a mean of −3.96 m/s against −1.44 observed. The trade wind is about 2.7 times too strong at the stations,
  consistent with an open-sea / low-roughness grid cell versus a sheltered 10 m anemometer.
- v: ERA5 is too strong (2.57 vs 1.17 m/s), has nearly twice the observed amplitude (1.54 vs 0.83 m/s) and its timing
  is weaker (anomaly correlation 0.66).

So the coastal gain is mainly a local bias and amplitude correction of a sea breeze that ERA5 already times correctly.
It is not the recovery of a missing breeze. State it this way in the text.

At l = 1 h (`ne_brazil_vit_l1`) the coastal model error is 0.568 m/s, against 0.689 for persistence (3.255 for ERA5).
GNN at 6 h: coastal 0.817, inland 1.197.

## E4: station density and availability filter

From `eval_fixed.csv` / `eval_ratio0.8.csv` (tables `e4_density.tex`, `e4_filter.tex`).

- Density: keep 1.0 / 0.75 / 0.5 / 0.25 of the 26 stations, seeds 0–2. Every network keeps the same 6 spread-out
  stations and every run is scored on those 6 (design decided 2026-11-08). At 0.25 the network is those 6 stations,
  so only seed 0 is trained.
- Filter: control ratio 0.6 / 0.7 / 0.8 (26 / 23 / 15 stations), all scored on the 15 stations kept at 0.8.

Density, on the 6 fixed stations; mean (sd) over seeds:

| fraction | stations | seeds | wind (m/s) | T RMSE (K) | Td RMSE (K) |
|---|---|---|---|---|---|
| 1.00 | 26 | 3 | 1.300 (0.017) | 0.868 (0.008) | 1.110 (0.014) |
| 0.75 | 20 | 3 | 1.305 (0.013) | 0.856 (0.010) | 1.114 (0.027) |
| 0.50 | 13 | 3 | 1.327 (0.004) | 0.863 (0.005) | 1.125 (0.032) |
| 0.25 | 6 | 1 | 1.334 | 0.876 | 1.107 |
| ERA5 | | | 1.838 | 1.229 | 1.757 |
| persistence | | | 2.359 | 4.606 | 2.630 |

Filter, on the 15 stations kept at 0.8 (95% CI):

| ratio | stations | wind (m/s) | T RMSE (K) | Td RMSE (K) |
|---|---|---|---|---|
| 0.6 | 26 | 1.064 [1.047, 1.080] | 1.031 [0.998, 1.061] | 1.333 [1.303, 1.361] |
| 0.7 | 23 | 1.065 [1.048, 1.081] | 1.043 [1.009, 1.074] | 1.315 [1.287, 1.341] |
| 0.8 | 15 | 1.079 [1.062, 1.095] | 1.053 [1.020, 1.083] | 1.337 [1.310, 1.362] |

Reading:
- **Density barely matters.** From 26 stations down to 6, wind error on the fixed stations rises monotonically but only
  from 1.300 to 1.334 m/s (+2.6%), about twice the training-seed sd (0.017 at 1.0) and each run's CI (about ±0.016).
  T and Td do not move beyond their spread. The degraded NE-US runs agree: thinning to 26 stations did not hurt there.
- So the model gains little from neighbouring stations at this spacing (~123 km). Its skill comes from ERA5 plus the
  station's own history. This is consistent with E3, where the gain is a per-station bias and amplitude correction.
- **The filter barely matters** either; 0.6 is at least as good as 0.8 on the stricter stations, which supports the 0.6
  choice (more stations, no cost in skill).
- Seed spread: the 1.0 row is three trainings of the full network (`ne_brazil_vit`, NumPy seed 100 with torch
  unseeded, and `ne_brazil_e4_frac1.0_tseed{1,2}` with `train_seed` 1 and 2). Its sd (wind 0.017) is pure training
  noise, the same size as the 0.75 / 0.5 sd, which mixes training noise and the station draw. So most of the seed
  spread is training noise, not which stations are kept.
