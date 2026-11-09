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

_To fill from `ne_brazil_{vit,gnn}/baselines.csv` (table `e2.tex`)._

| forecast | wind-vector error (m/s) | T RMSE (K) | Td RMSE (K) |
|---|---|---|---|
| Transformer + ERA5 Nearest | | | |
| GNN (Delaunay) + ERA5 grid | | | |
| ERA5 nearest | | | |
| persistence | | | |

At l = 1 h persistence is a near-ceiling on NE-US (0.403 m/s wind vs the model's 0.401, and better on Td), which is
why the headline lead is 6 h. The model–persistence gap is the number to discuss.

## E3: sea breeze (main result)

_To fill from `ne_brazil_vit/diurnal/` (`wind_error.png`, `uv_cycle.png`, table `e3.tex`)._

- Coastal = within 20 km of the Natural Earth 10 m coastline (decided 2026-11-08). Stations per group: coastal _n_,
  inland _n_ (the script also logs the counts at 10/30/50 km).
- Local time UTC−3.
- Reading the figure: ERA5 error is discontinuous at 09/10 and 21/22 UTC (06/07 and 18/19 local), the boundaries of
  ERA5's 12 h assimilation windows; seen on NE-US too, not a pipeline artefact.

## E4: station density and availability filter

_To fill from `eval_fixed.csv` / `eval_ratio0.8.csv` (tables `e4_density.tex`, `e4_filter.tex`)._

- Density: keep 1.0 / 0.75 / 0.5 / 0.25 of the 26 stations, seeds 0–2. Every network keeps the same 6 spread-out
  stations and every run is scored on those 6 (design decided 2026-11-08). At 0.25 the network is those 6 stations,
  so only seed 0 is trained.
- Filter: control ratio 0.6 / 0.7 / 0.8 (26 / 23 / 15 stations), all scored on the 15 stations kept at 0.8.

## Lead sweep (Transformer only)

_To fill from `ne_brazil_vit{,_l1,_l24,_l48}/baselines.csv` (table `leads_brazil.tex`)._

| lead (h) | model wind | ERA5 wind | persistence wind | model T RMSE | persistence T RMSE |
|---|---|---|---|---|---|
| 1 | | | | | |
| 6 | | | | | |
| 24 | | | | | |
| 48 | | | | | |
