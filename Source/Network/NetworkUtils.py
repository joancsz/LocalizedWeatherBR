# Author: Qidong Yang & Jonathan Giezendanner

import torch


def search_k_neighbors(target_points, source_points, k):
    # base_points: (n_b, n_features) : where the points go to
    # cand_points: (n_c, n_features): where the points come from

    # brute force instead of torch_geometric knn, which needs pyg-lib/torch-cluster
    # shortcut: dense n_b x n_c distances, fine for ERA5 boxes; chunk it if used on the full HRRR grid
    nearest = torch.cdist(target_points, source_points).topk(k, dim=1, largest=False).indices
    edge_index = torch.stack([nearest.flatten(), torch.arange(len(target_points)).repeat_interleave(k)])

    # edge_index: 0: source, 1: target

    return edge_index
