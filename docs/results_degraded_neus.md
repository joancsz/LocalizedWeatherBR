# Results: degraded NE-US (PLAN_UPDATE_v2 §2.5)

Source: `notebooks/03_degraded_neus.ipynb` (Colab GPU). LaTeX: `$DATA_ROOT/tables/degraded.tex`.

Goal: separate the effect of a sparse network, station outages and a coarse grid from the effect of the tropical
climate, using the paper's NE-US data. Transformer + ERA5 Nearest, lead 1 h, 20 epochs, `use_is_real_mask: true` in
every run (so `base` is not the `neus_vit_20` run of `docs/repro_log.md`, which trained on filled values too).
These are reanalysis downscaling results, as are the paper's ERA5 / HRRR-A numbers.

| run | stations | outages | ERA5 grid | wind-vector error (m/s) | T RMSE (K) | Td RMSE (K) |
|---|---|---|---|---|---|---|
| base | 358 | none | 0.25° | | | |
| sparse | 26 (spread, seed 0) | none | 0.25° | | | |
| gaps | 358 | Brazil-like | 0.25° | | | |
| coarse_100 | 358 | none | 1.0° | | | |
| all | 26 | Brazil-like | 1.0° | | | |
| sparse_gaps | 26 | Brazil-like | 0.25° | | | |
| coarse_050 | 358 | none | 0.5° | | | |

- Outage parameters: `target_real_fraction` _x_ and `mean_gap_hours` _x_, measured in notebook 03 on the 26 Brazil
  stations (fraction of real `u` hours, mean outage length).
- Outages remove whole station-hours (all variables) and are drawn per year, test year included, so the gap runs are
  scored on fewer test hours (the notebook's table lists the real fraction per run).
- **Which row compares with Brazil:** `sparse_gaps`. Brazil's ERA5 is also 0.25°, so `all` (1.0°) is harsher than
  the Brazil setting; it bounds the grid effect rather than mimicking Brazil.
