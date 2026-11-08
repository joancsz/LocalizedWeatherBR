"""PLAN_UPDATE_v2 §2.5 (1): subsample NE-US to n spatially spread stations.

Interface pinned here (to implement in Source/Utils/Degrade.py and Source/Utils/Paths.py):
  Degrade.spread_stations(lons, lats, n, seed) -> np.ndarray of row indices into the filtered shapefile
  Paths.filtered_file_name(file_name, control_ratio, keep_n=None, seed=None)
"""
import numpy as np
import pytest

Degrade = pytest.importorskip('Utils.Degrade', reason='station_keep_n not implemented yet (PLAN_UPDATE_v2 §2.5)')
import Utils.Paths as Paths


def grid_with_cluster():
    # 10x10 regular grid plus a dense cluster of 50 stations in one corner (like NE-US around Boston/NYC)
    g = np.stack(np.meshgrid(np.linspace(-80, -67, 10), np.linspace(39, 47, 10)), -1).reshape(-1, 2)
    rng = np.random.default_rng(0)
    cluster = np.array([-71.0, 42.3]) + rng.normal(0, 0.05, (50, 2))
    pts = np.vstack([g, cluster])
    return pts[:, 0], pts[:, 1]


def test_exactly_n_unique_valid_indices():
    lons, lats = grid_with_cluster()
    idx = Degrade.spread_stations(lons, lats, 26, seed=0)
    assert len(idx) == 26
    assert len(np.unique(idx)) == 26
    assert idx.min() >= 0 and idx.max() < len(lons)


def test_same_seed_same_selection():
    lons, lats = grid_with_cluster()
    a = Degrade.spread_stations(lons, lats, 26, seed=3)
    b = Degrade.spread_stations(lons, lats, 26, seed=3)
    np.testing.assert_array_equal(a, b)


def test_different_seed_changes_selection():
    lons, lats = grid_with_cluster()
    assert any(set(Degrade.spread_stations(lons, lats, 26, seed=s)) != set(Degrade.spread_stations(lons, lats, 26, seed=0))
               for s in range(1, 5))


def test_spread_not_clustered():
    # farthest-point / stratified picks at most a couple from the 50-station cluster; pure random would take ~1/3
    lons, lats = grid_with_cluster()
    idx = Degrade.spread_stations(lons, lats, 26, seed=0)
    assert np.sum(idx >= 100) <= 3


def test_n_equal_to_all_keeps_all():
    lons, lats = grid_with_cluster()
    assert sorted(Degrade.spread_stations(lons, lats, len(lons), seed=0)) == list(range(len(lons)))


def test_filename_has_n_and_seed_and_default_unchanged():
    base = Paths.filtered_file_name('stem', 0.9)
    assert base == Paths.filtered_file_name('stem', 0.9, keep_n=None, seed=None)  # reproduction must not drift
    a = Paths.filtered_file_name('stem', 0.9, keep_n=26, seed=0)
    b = Paths.filtered_file_name('stem', 0.9, keep_n=26, seed=1)
    assert len({base, a, b}) == 3
    assert '26' in a and a.startswith(base)


def test_keep_fraction_always_contains_fixed_eval_set():
    lons, lats = grid_with_cluster()
    ev = set(Degrade.eval_stations(lons, lats, 6))
    for frac in (1.0, 0.75, 0.5, 0.25):
        for seed in (0, 1, 2):
            idx = Degrade.keep_fraction(lons, lats, frac, 6, seed)
            assert len(idx) == max(round(frac * len(lons)), 6) and len(set(idx)) == len(idx)
            assert ev <= set(idx)
            assert list(idx) == sorted(idx)                 # shapefile row order (I1)
    assert set(Degrade.keep_fraction(lons, lats, 0.5, 6, 0)) != set(Degrade.keep_fraction(lons, lats, 0.5, 6, 1))
    np.testing.assert_array_equal(Degrade.keep_fraction(lons, lats, 0.5, 6, 1), Degrade.keep_fraction(lons, lats, 0.5, 6, 1))


def test_brazil_quarter_is_the_eval_set():
    # 26 stations at 25% -> round(6.5) = 6 = the evaluation set itself
    rng = np.random.default_rng(0)
    lons, lats = rng.uniform(-41, -35, 26), rng.uniform(-9, -3, 26)
    assert set(Degrade.keep_fraction(lons, lats, 0.25, 6, 2)) == set(Degrade.eval_stations(lons, lats, 6))


def test_fraction_filename():
    a = Paths.filtered_file_name('stem', 0.6, seed=1, keep_fraction=0.5, eval_n=6)
    assert a == 'stem_filtered_0.6_frac0.5_eval6_seed1'
