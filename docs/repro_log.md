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
| GNN (Delaunay) + ERA5 grid, 20 ep, lead 1 h | pending | pending | pending |

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
