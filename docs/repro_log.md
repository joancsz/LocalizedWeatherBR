# Reproduction log

## P1: old vs new first-epoch loss (2026-10-24)

Setup: NE-US Zenodo data at `data/`, Transformer (ViT), ERA5 Nearest, `n_neighbors_e2m=1`, `lead_hrs=1`,
`madis_control_ratio=0.9`, 1 epoch, CPU (torch 2.14.1+cpu, Python 3.12.10), `torch.manual_seed(0)` set before
`Main.Run()` (which itself sets `np.random.seed(100)`). 358 stations.

| code | train_loss (epoch 1) | valid_loss (epoch 1) |
|---|---|---|
| unmodified `a0329a5` | 1.26501 | 0.79431 |
| phase 1 (`configs/neus_repro.yaml`, `use_is_real_mask: false`) | 1.26501 | 0.79431 |

Commands: a scratch harness built the old `LocalLauncher` args (with `model_type=ViT`) for the base worktree, and
`run.load_config('configs/neus_repro.yaml')` for the new code, both with `epochs=1`.

Test-set `IsReal` (from `IsReal_CUSTOM_min.pkl`, shape matches `Preds`, 8711×358×4): real fraction
u/v/T = 0.964, Td = 0.866. So without `use_is_real_mask`, about 4% of wind/T targets and 13% of Td targets are filled
values.
