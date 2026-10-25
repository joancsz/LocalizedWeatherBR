from pathlib import Path

import pytest

import Utils.Paths as Paths
from LocalLauncher import AttrDict
from PostProcessInputs import PostProcessArgs
from run import DEFAULTS

NEUS = 'stations_2019_2023_-80.53_-66.94_38.92_47.47northeastern_buffered_filtered_0.9.shp'


def test_names_match_zenodo_layout():
    name = Paths.region_file_name(-80.53, -66.94, 38.92, 47.47, 'Shapefiles/Regions/northeastern_buffered.shp')
    filtered = Paths.filtered_file_name(name, 0.9)
    assert Paths.station_file('d', 2019, 2023, filtered) == Path('d/madis/stations') / NEUS
    assert Paths.madis_processed_file('d', 2019, 2023, 2021, filtered) == \
        Path('d/madis/processed/Meta-2019-2023') / f'madis_2021_{filtered}.nc'
    assert Paths.era5_monthly_file('d', 'World', 2019, 3) == Path('d/ERA5/World/surface_2019_3.nc')


def args(**kw):
    a = AttrDict(DEFAULTS)
    a.update(kw)
    return PostProcessArgs(a)


def test_default_split_is_legacy():
    a = args()
    assert (a.start_year, a.n_years) == (2019, 5)
    assert (a.train_years, a.val_years, a.test_years) == ([2019, 2020, 2021], [2022], [2023])


def test_explicit_split_validated():
    a = args(start_year=2018, end_year=2023, train_years=[2018, 2019], val_years=[2020], test_years=[2023])
    assert a.n_years == 6
    with pytest.raises(AssertionError):
        args(train_years=[2019, 2020], val_years=[2020], test_years=[2023])
    with pytest.raises(AssertionError):
        args(train_years=[2015], val_years=[2022], test_years=[2023])
