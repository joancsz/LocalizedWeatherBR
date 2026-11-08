import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'eval'))
import station_density


def test_match_by_rounded_coordinates_in_eval_order():
    run_lon, run_lat = np.array([-40.1, -38.123449, -36.0]), np.array([-5.0, -6.5, -7.0])
    idx = station_density.match_stations(run_lon, run_lat, np.array([-36.0, -38.12345]), np.array([-7.0, -6.5]))
    assert list(idx) == [2, 1]
    with pytest.raises(ValueError):
        station_density.match_stations(run_lon, run_lat, np.array([-30.0]), np.array([-7.0]))
