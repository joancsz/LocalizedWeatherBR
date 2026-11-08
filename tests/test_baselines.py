import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'eval'))
import baselines


def ds(offset):
    # value encodes (station, hour) so a wrong time or station pick is visible
    time = pd.date_range('2023-01-01', periods=10, freq='h')
    val = offset + 100 * np.arange(1, 4)[:, None] + np.arange(10)[None, :]
    return xr.Dataset({n: (('stations', 'time'), val + k * 1000.) for k, n in enumerate(baselines.NAMES)},
                      coords=dict(stations=[1, 2, 3], time=time))


def test_alignment_with_sample_times():
    back, lead = 2, 3
    t = pd.date_range('2023-01-01', periods=10, freq='h').values.astype('datetime64[us]').astype(np.int64)
    times = np.stack([t[i:i + back + lead + 1] for i in range(10 - back - lead)])  # like MixData samples
    out = baselines.baseline_preds(times, lead, ds(0.), ds(0.5), [0, 1, 2, 3])
    assert out['ERA5'].shape == out['persistence'].shape == (len(times), 3, 4)
    # sample 0: issue hour index back=2, target hour index back+lead=5
    assert out['persistence'][0, 1, 0] == 200 + 2
    assert out['ERA5'][0, 1, 0] == 200 + 5 + 0.5
    assert out['ERA5'][1, 2, 3] == 3000 + 300 + 6 + 0.5      # dewpoint, third station, next sample


def test_ns_and_us_times_agree():
    t = pd.date_range('2023-06-01', periods=3, freq='h')
    us = t.values.astype('datetime64[us]').astype(np.int64)
    ns = t.values.astype('datetime64[ns]').astype(np.int64)
    assert (baselines.to_datetime(us) == baselines.to_datetime(ns)).all()


def test_output_var_order_follows_run():
    t = pd.date_range('2023-01-01', periods=10, freq='h').values.astype('datetime64[us]').astype(np.int64)
    times = t[None, :4]
    out = baselines.baseline_preds(times, 1, ds(0.), ds(0.), [2, 0])   # temp, u
    assert out['persistence'][0, 0, 0] == 2000 + 100 + 2 and out['persistence'][0, 0, 1] == 100 + 2
