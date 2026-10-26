import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'eval'))
import metrics


def test_hand_computed():
    # one sample, two stations, vars (u, v, temp, dewpoint)
    preds = np.array([[[3., 4., 290., 280.], [0., 0., 300., 285.]]])
    targets = np.array([[[0., 0., 291., 280.], [0., 0., 296., 99.]]])
    is_real = np.array([[[True, True, True, True], [True, False, True, False]]])

    all_rows, real = metrics.summarize(preds, targets, is_real, [0, 1, 2, 3])
    assert np.isclose(all_rows['wind'], 2.5)          # mean(5, 0)
    assert np.isclose(real['wind'], 5.0)              # station 2 has v filled
    assert np.isclose(real['t_rmse'], np.sqrt((1 + 16) / 2))
    assert np.isclose(real['t_bias'], (-1 + 4) / 2)
    assert np.isclose(real['td_rmse'], 0.0)           # filled Td at station 2 excluded
    assert all_rows['td_rmse'] > 100
