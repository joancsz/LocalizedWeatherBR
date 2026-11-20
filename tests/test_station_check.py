import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'eval'))
import station_check as sc


def test_longest_run():
    assert sc.longest_run(np.array([1, 1, 2, 2, 2, 3])) == 3
    assert sc.longest_run(np.array([5])) == 1
    assert sc.longest_run(np.array([])) == 0


def test_obs_flags_only_real_pairs():
    obs = np.array([300., 300., 310., 301., 301.])
    real = np.array([True, True, False, True, True])
    f = sc.obs_flags(obs, real)                      # the 300->310->301 jumps touch a non-real hour
    assert f['jumps_gt5'] == 0 and f['n_real'] == 4 and f['longest_stuck_h'] == 2
