# Reproduction log

## P1: old vs new first-epoch loss (2026-10-24)

Setup: NE-US Zenodo data at `data/`, Transformer (ViT), ERA5 Nearest, `n_neighbors_e2m=1`, `lead_hrs=1`,
`madis_control_ratio=0.9`, 1 epoch, CPU (torch 2.14.1+cpu, Python 3.12.10), `torch.manual_seed(0)` set before
`Main.Run()` (which itself sets `np.random.seed(100)`). 358 stations.

| code | train_loss (epoch 1) | valid_loss (epoch 1) |
|---|---|---|
| unmodified `a0329a5` | 1.26501 | 0.79431 |
| phase 1 (`configs/neus_repro.yaml`, `use_is_real_mask: false`) | 1.26501 | 0.79431 |

2-epoch run (plan acceptance), same setup: one CPU epoch takes ~12 h (the old 2-epoch run took 1475 min).

| code | train_loss ep1 | train_loss ep2 | valid_loss ep2 |
|---|---|---|---|
| unmodified `a0329a5` | 1.26501 | 0.66544 | 0.57957 |
| phase 1 | 1.26501 | pending | pending |

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

Both: ERA5 Nearest, `n_neighbors_e2m=1`, `lead_hrs=1`, 20 epochs, real-only metrics. The GNN uses the DELAUNAY graph.
The paper averages over leads and trains for 200 epochs; the gap is recorded, not tuned away.

| model | wind-vector error (m/s) | T RMSE (°C) | Td RMSE (°C) |
|---|---|---|---|
| paper, Transformer + ERA5 | 0.50 | 1.44 | 1.55 |
| Transformer + ERA5, 20 ep | pending | pending | pending |
| GNN (Delaunay) + ERA5, 20 ep | pending | pending | pending |
