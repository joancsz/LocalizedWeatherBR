"""PLAN_UPDATE_v2 §2.5 (3): block-mean coarsening of ERA5 before interpolation to stations.

Interface pinned here (to implement in Source/Utils/Degrade.py):
  Degrade.coarsen_era5(ds, factor) -> xarray.Dataset, block mean over (latitude, longitude)
The NE-US processed ERA5 grid is 49 x 73 (not divisible by 2 or 4), so edges are trimmed.
"""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

Degrade = pytest.importorskip('Utils.Degrade', reason='era5_coarsen_deg not implemented yet (PLAN_UPDATE_v2 §2.5)')

VARS = ['u10', 'v10', 't2m', 'd2m', 'ssr']


def era5(n_lat=48, n_lon=72):
    # same layout as data/ERA5/Processed/era5_{year}_*.nc: 0.25 deg, latitude descending (I5)
    lat = 49.0 - 0.25 * np.arange(n_lat)
    lon = -83.0 + 0.25 * np.arange(n_lon)
    time = pd.date_range('2019-01-01', periods=6, freq='h')
    rng = np.random.default_rng(0)
    return xr.Dataset({v: (('time', 'latitude', 'longitude'), rng.normal(size=(6, n_lat, n_lon))) for v in VARS},
                      coords=dict(time=time, latitude=lat, longitude=lon))


@pytest.mark.parametrize('factor', [2, 4])
def test_cells_per_axis_divided(factor):
    out = Degrade.coarsen_era5(era5(), factor)
    assert out.sizes['latitude'] == 48 // factor and out.sizes['longitude'] == 72 // factor
    assert out.sizes['time'] == 6
    assert np.allclose(np.diff(out.latitude), -0.25 * factor)   # 0.5 / 1.0 deg spacing
    assert np.allclose(np.diff(out.longitude), 0.25 * factor)


@pytest.mark.parametrize('factor', [2, 4])
def test_domain_mean_preserved(factor):
    ds = era5()
    out = Degrade.coarsen_era5(ds, factor)
    for v in VARS:
        np.testing.assert_allclose(out[v].mean(('latitude', 'longitude')), ds[v].mean(('latitude', 'longitude')))


def test_block_mean_value():
    ds = era5()
    out = Degrade.coarsen_era5(ds, 2)
    np.testing.assert_allclose(out.t2m.isel(time=0, latitude=0, longitude=0),
                               ds.t2m.isel(time=0, latitude=slice(0, 2), longitude=slice(0, 2)).mean())


def test_latitude_stays_descending_and_vars_kept():
    out = Degrade.coarsen_era5(era5(), 2)
    assert out.latitude[0] > out.latitude[-1]                     # I5, the loader slices lat_up..lat_low
    assert set(VARS) <= set(out.data_vars)
    assert not any(bool(out[v].isnull().any()) for v in VARS)       # I4


def test_real_neus_shape_trims_edges():
    # 49 x 73 like the Zenodo file: factor 4 -> 12 x 18, no NaN cells from partial blocks
    out = Degrade.coarsen_era5(era5(49, 73), 4)
    assert (out.sizes['latitude'], out.sizes['longitude']) == (12, 18)
    assert not bool(out.t2m.isnull().any())


def test_factor_one_is_identity():
    ds = era5()
    xr.testing.assert_allclose(Degrade.coarsen_era5(ds, 1), ds)


def test_grid_tag_keeps_native_cache_names():
    import Utils.Paths as Paths
    assert Paths.era5_grid_tag(0.25) == ''          # existing 0.25 deg caches keep their names
    assert Paths.era5_grid_tag(0.5) == '_c050'
    assert Paths.era5_grid_tag(1.0) == '_c100'
