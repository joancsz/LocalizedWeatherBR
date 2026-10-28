# Reproduction log

## P1: old vs new first-epoch loss (2026-10-24)

Setup: NE-US Zenodo data at `data/`, Transformer (ViT), ERA5 Nearest, `n_neighbors_e2m=1`, `lead_hrs=1`,
`madis_control_ratio=0.9`, 1 epoch, CPU (torch 2.14.1+cpu, Python 3.12.10), `torch.manual_seed(0)` set before
`Main.Run()` (which itself sets `np.random.seed(100)`). 358 stations.

| code | train_loss (epoch 1) | valid_loss (epoch 1) |
|---|---|---|
| unmodified `a0329a5` | 1.26501 | 0.79431 |
| phase 1 (`configs/neus_repro.yaml`, `use_is_real_mask: false`) | 1.26501 | 0.79431 |

2-epoch run (plan acceptance), same setup: the old 2-epoch run took 1475 min of wall time and the new one 687 min (same machine, run one after the other).

| code | train_loss ep1 | train_loss ep2 | valid_loss ep2 |
|---|---|---|---|
| unmodified `a0329a5` | 1.26501 | 0.66544 | 0.57957 |
| phase 1 | 1.26501 | 0.66544 | 0.57957 |

Commands: a scratch harness built the old `LocalLauncher` args (with `model_type=ViT`) for the base worktree, and
`run.load_config('configs/neus_repro.yaml')` for the new code, both with `epochs=1`.

Test-set `IsReal` (from `IsReal_CUSTOM_min.pkl`, shape matches `Preds`, 8711×358×4): real fraction
u/v/T = 0.964, Td = 0.866. So without `use_is_real_mask`, about 4% of wind/T targets and 13% of Td targets are filled
values.

Test metrics of the 1-epoch phase 1 run (`python scripts/eval/metrics.py --run-dir data/ModelOutputs/compare_new`).
This is only a sanity check of the metrics script; 1 epoch is far from converged:

| entries | wind-vector error (m/s) | T RMSE (K) | Td RMSE (K) |
|---|---|---|---|
| all | 0.763 | 2.482 | 2.756 |
| real only | 0.760 | 2.404 | 2.272 |

## P2: NE-US 20-epoch runs (Colab GPU)

Commands (`notebooks/colab_setup.ipynb`, `DATA_ROOT=/content/drive/MyDrive/LocalizedWeather`):

```
python Source/run.py --config configs/neus_vit_20.yaml
python Source/run.py --config configs/neus_gnn_20.yaml
python scripts/eval/metrics.py --run-dir $DATA_ROOT/neus/ModelOutputs/neus_vit_20
python scripts/eval/metrics.py --run-dir $DATA_ROOT/neus/ModelOutputs/neus_gnn_20
```

Transformer: ERA5 Nearest, `n_neighbors_e2m=1`. GNN: DELAUNAY graph, raw ERA5 grid nodes (`interpolation_type=-1`,
`n_neighbors_e2m=8`). Both: `lead_hrs=1`, 20 epochs, metrics on real observations only. The paper averages over leads
and trains for 200 epochs; the gap is recorded, not tuned away.

| model | wind-vector error (m/s) | T RMSE (°C) | Td RMSE (°C) |
|---|---|---|---|
| paper, Transformer + ERA5 (avg. over leads) | 0.50 | 1.44 | 1.55 |
| Transformer + ERA5 Nearest, 20 ep, lead 1 h | 0.399 | 0.953 | 0.962 |
| GNN (Delaunay) + ERA5 grid, 20 ep, lead 1 h | 0.574 | 2.278 | 1.633 |

Transformer run (2026-10-26, Colab A100, torch 2.14.1): train 1.24706 → 0.32974, best val 0.33702 at epoch 19 (the
checkpoint used for test); val was still falling, so 20 epochs is not converged. All-entries metrics: 0.389 / 0.966 / 0.932.
The 1 h lead is far easier than the paper's average over leads, which is why it beats the paper; only the
multi-lead sweep (E5) gives a like-for-like comparison. Real-only Td RMSE is *higher* than all-entries (0.962 vs 0.932),
because filled values are forward-filled and therefore easy to predict: evaluate on real observations only.
No torch seed is set in `Main`, so the Colab epoch-1 loss (1.24706) differs from the seeded local runs (1.26501).

First GNN attempt crashed at the first batch (`shape '[128, 358, 1]' is invalid for input of size 128` in
`MPNN.build_graph_external`): with an interpolated ERA5 no external network is built, so `MixData` passes placeholder
positions. Same behaviour in the original code. Fixed in the config (`interpolation_type: -1`, `n_neighbors_e2m: 8`),
and `Main` now asserts this combination up front.

GNN run (2026-10-26, Colab L4, torch 2.14.1, after the fix): train 3.55598 → 0.53493, best val 0.56589 at epoch 20 (the
last epoch, used for test); val was still falling steeply, so the GNN is further from convergence than the Transformer.
All-entries metrics: 0.578 / 2.335 / 1.743. Its T RMSE is worse than the paper's lead-averaged number; this is
under-training at 20 epochs, not tuned. Metrics re-computed locally from the downloaded `ModelOutputs` match the notebook output.

## P3: NE Brazil data pipeline (in progress)

Run order: `notebooks/02_brazil_data.ipynb`. Deviations from `PLAN_BRAZIL.md` §4, and why:

- **Region source**: IBGE malhas API (the plan's fallback), not `geobr`. Same 2020 polygons, no extra dependency.
  The API gzips its response unconditionally, so the script decompresses before parsing.
- **INMET download**: `portal.inmet.gov.br` resets scripted connections from the dev PC (Zenodo and IBGE work from
  the same machine), and the scripted download also failed on Colab. Per the plan, no mirror was tried; the 2018–2025 zips were downloaded by hand
  into `$DATA_ROOT/brazil/inmet/zips/`, which `download_inmet.py` extracts without downloading.
- **ERA5 area**: the plan's "region bounds + 2°" does not cover the loader's slice, which is
  `floor(low − 1.5) .. ceil(up + 1.5)` = lon [−44, −33], lat [−12, −1] (bounds + 2° stops at lon −43.52).
  `download_era5.py` requests exactly that integer box. The new CDS returns a zip (instant + accumulated streams)
  for this variable mix; the script merges it into one NetCDF per month.
- **Filtered files**: the adapter writes the all-station shapefile and all-station processed NetCDFs; the filtered
  shapefile comes from `MetaStation` and the filtered NetCDFs from `Madis.createFile`'s selection, so station order
  (I1) is the training code's own. Another `madis_control_ratio` needs no rerun.
- **INMET 2019 format** (589 files, all parse): metadata keys have literal `?` for accented letters, missing values
  are empty strings (no `-9999`), one station (A302) writes latitude `,916877`.

Single-year check on 2019 (dev PC, `--years 2019`): 48 of 589 stations inside the region; stations kept per
control ratio 0.6/0.7/0.8/0.9 = 44/43/35/25. Diurnal temperature peak (UTC) across stations: 15 h ×7, 16 h ×10,
17 h ×17, 18 h ×14, median 17 UTC, so timestamps are UTC. Real fraction at 0.8: u/v/T 0.94, Td 0.92, solar 0.52
(night hours are blank). The repo's `Madis` loads the filtered file; I1, I2 and I4 hold. Counts for 2019–2023 come
from the full run.

ERA5 check on 2019 (12 monthly CDS files, about 4–5 min queue each): the repo's `ERA5Interpolated` (Nearest,
`era5_region: NEBrazil`) builds the interpolated file for the 35 stations, 8760 hours, no NaN. Median diurnal
temperature peak is 17 UTC for both INMET and ERA5-nearest (I3); 2019 station-mean T is 299.54 K (INMET) vs 299.97 K (ERA5).
