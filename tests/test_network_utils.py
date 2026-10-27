import torch

from Network.NetworkUtils import search_k_neighbors


def test_k_nearest_sources_per_target():
    sources = torch.tensor([[0., 0.], [1., 0.], [5., 5.], [0., 2.]])
    targets = torch.tensor([[0.1, 0.], [4.9, 5.]])
    edge_index = search_k_neighbors(targets, sources, 2)
    assert edge_index.shape == (2, 4)
    assert edge_index[1].tolist() == [0, 0, 1, 1]          # row 1: target, k per target
    assert edge_index[0, :2].tolist() == [0, 1]             # row 0: nearest sources first
    assert edge_index[0, 2].item() == 2
