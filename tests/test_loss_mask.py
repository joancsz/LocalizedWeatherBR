import torch

from Settings.Settings import EnvVariables, LossFunctionType, LossFunctionType as L
from Utils.LossFunctions import GetLossFunction, GetLossFunctionReport, SetupSaveMetrics

VARS = [EnvVariables.u, EnvVariables.v, EnvVariables.temp, EnvVariables.dewpoint]


def make_batch():
    torch.manual_seed(0)
    out = torch.randn(2, 5, 4)
    y = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, 4, dtype=torch.bool)
    mask[0, 1, :] = False       # whole station missing
    mask[1, 3, 2] = False       # temp missing only
    mask[1, 4, 1] = False       # v missing -> wind entry dropped
    return out, y, mask


def test_mask_off_is_unchanged():
    out, y, mask = make_batch()
    masked_off = GetLossFunction(L.CUSTOM, VARS)(out, y, mask)
    all_real = GetLossFunction(L.CUSTOM, VARS, True)(out, y, torch.ones_like(mask))
    assert torch.allclose(masked_off, all_real)


def test_mask_ignores_filled_entries():
    out, y, mask = make_batch()
    loss = GetLossFunction(L.CUSTOM, VARS, True)
    base = loss(out, y, mask)

    corrupted = y.clone()
    corrupted[~mask] = 1e6  # filled values must not reach the loss
    corrupted[1, 4, 0] = 1e6  # u is real but v is not, so the wind entry is dropped
    assert torch.allclose(loss(out, corrupted, mask), base)

    report = GetLossFunctionReport(L.CUSTOM, VARS, True)
    save = SetupSaveMetrics([LossFunctionType.CUSTOM], VARS, True)[LossFunctionType.CUSTOM]
    assert torch.allclose(report(out, corrupted, mask), report(out, y, mask))
    assert torch.allclose(save(out, corrupted, mask), save(out, y, mask))


def test_masked_mean_matches_manual():
    out, y, mask = make_batch()
    err = (out - y) ** 2
    eps = torch.finfo(torch.float32).eps
    wind = torch.sqrt(err[..., 0] + err[..., 1] + eps)[mask[..., 0] & mask[..., 1]]
    other = torch.sqrt(err[..., 2:] + eps)[mask[..., 2:]]
    expected = torch.cat([wind, other]).mean()
    assert torch.allclose(GetLossFunction(L.CUSTOM, VARS, True)(out, y, mask), expected)
