import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import LineString

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'eval'))
import diurnal_analysis as da


def us(ts):
    return pd.DatetimeIndex(ts).values.astype('datetime64[us]').astype(np.int64)


def test_local_hour_from_target_column():
    t = us(['2023-01-01 02:00', '2023-01-01 03:00', '2023-01-01 15:00'])
    times = np.stack([t - 3600 * 10**6, t], axis=1)         # last column is the target hour
    assert list(da.local_hours(times, -3)) == [23, 0, 12]


def test_coast_distance_km():
    # coastline along the meridian -35; stations 0.1 and 1 degree west of it at ~6 S (1 deg lon ~ 110.6 km there)
    coast = gpd.GeoDataFrame(geometry=[LineString([(-35, -10), (-35, -2)])], crs='EPSG:4326')
    d = da.coast_distance_km(np.array([-35.1, -36.0]), np.array([-6.0, -6.0]), coast)
    np.testing.assert_allclose(d, [11.06, 110.6], rtol=0.02)


def test_wind_error_by_hour_groups_and_mask():
    n_days, n_st = 10, 3
    hours = np.tile(np.arange(24), n_days)
    days = np.repeat(np.arange(n_days), 24)
    targets = np.zeros((240, n_st, 4))
    model = targets.copy()
    model[..., 0] = 3.0                                     # u error 3 everywhere
    model[hours == 14, 0, 0] = 5.0                          # station 0 worse at 14 h
    era5 = targets.copy()
    era5[..., 1] = 4.0
    is_real = np.ones_like(targets, bool)
    is_real[:, 2, 1] = False                                # station 2 never has real v -> excluded from wind
    fc = {'model': model, 'ERA5': era5, 'persistence': targets}
    groups = {'coastal': np.array([True, False, False]), 'inland': np.array([False, True, True])}
    df = da.wind_error_by_hour(fc, targets, is_real, [0, 1, 2, 3], hours, days, groups).set_index(
        ['group', 'forecast', 'hour'])
    assert df.loc[('coastal', 'model', 14), 'wind'] == 5.0
    assert df.loc[('coastal', 'model', 13), 'wind'] == 3.0
    assert df.loc[('inland', 'ERA5', 0), 'wind'] == 4.0
    assert df.loc[('inland', 'model', 0), 'n'] == n_days     # only station 1 counts
    assert df.loc[('coastal', 'persistence', 5), 'wind'] == 0.0


def test_uv_cycle_uses_same_entries():
    hours = np.tile(np.arange(24), 2)
    targets = np.zeros((48, 1, 4))
    targets[hours == 12, 0, 0] = 2.0
    is_real = np.ones_like(targets, bool)
    fc = {'model': targets + 1, 'ERA5': targets - 1}
    df = da.uv_cycle(fc, targets, is_real, [0, 1, 2, 3], hours, {'coastal': np.array([True])})
    row = df[(df.hour == 12)].set_index('source')
    assert row.loc['observed', 'u'] == 2.0 and row.loc['model', 'u'] == 3.0 and row.loc['ERA5', 'u'] == 1.0
