import torch

from Modules.Transformer.ViT import VisionTransformer


def _outputs(isolated):
    torch.manual_seed(0)
    model = VisionTransformer(5, 8, 4, 4, 16, 8, 16, 3, 2, era5_n_vars=3, era5_len=8,
                              station_isolated=isolated).eval()
    madis, era5 = torch.randn(2, 5, 8, 4), torch.randn(2, 5, 8, 3)
    base = model(madis, era5)[0]
    madis[:, 1:] += 10.0  # change every station except station 0
    era5[:, 1:] += 10.0
    return base[:, 0], model(madis, era5)[0][:, 0]


def test_isolated_station_ignores_other_stations():
    before, after = _outputs(True)
    assert torch.allclose(before, after, atol=1e-5)


def test_default_station_sees_other_stations():
    before, after = _outputs(False)
    assert not torch.allclose(before, after, atol=1e-5)
