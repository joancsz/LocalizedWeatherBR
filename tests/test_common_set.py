import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'eval'))
import common_set


def make_runs():
    rng = np.random.default_rng(0)
    xy = rng.random((5, 2))
    times = np.arange(10)
    targets = rng.random((10, 5, 4))
    real = np.ones((10, 5, 4), bool)
    a = (xy, times, targets + 0.1, targets, real)
    perm, rev = np.array([4, 2, 0, 3, 1]), np.arange(10)[::-1]   # b: other station order, reversed hours, one gap
    real_b = real.copy()
    real_b[3, 0] = False                                        # hour 3, station 0 (in a's order)
    b = (xy[perm], times[rev], (targets - 0.2)[rev][:, perm], targets[rev][:, perm], real_b[rev][:, perm])
    return {'a': a, 'b': b}, xy[[0, 3]], targets


def test_align_matches_stations_hours_and_joint_mask():
    runs, ref, targets = make_runs()
    out, mask, common = common_set.align(runs, ref)
    assert list(common) == list(range(10))
    np.testing.assert_allclose(out['a'][1], targets[:, [0, 3]])
    np.testing.assert_allclose(out['b'][1], targets[:, [0, 3]])
    np.testing.assert_allclose(out['b'][0], targets[:, [0, 3]] - 0.2)
    assert not mask[3, 0].any() and mask.sum() == mask.size - 4  # b's gap, back in common hour order


def test_target_mismatch_and_missing_station_raise():
    runs, ref, _ = make_runs()
    xy, t, p, tg, r = runs['b']
    with pytest.raises(ValueError, match='targets differ'):
        common_set.align({**runs, 'b': (xy, t, p, tg + 1, r)}, ref)
    with pytest.raises(ValueError, match='missing'):
        common_set.align(runs, np.array([[9.0, 9.0]]))
