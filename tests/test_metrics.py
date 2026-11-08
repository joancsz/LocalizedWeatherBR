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


def test_ci_degenerate_and_contains_estimate():
    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(30), 24)
    const = np.full((720, 5), 2.0)
    assert metrics.day_bootstrap_ci(const, None, days) == (2.0, 2.0)
    assert np.allclose(metrics.day_bootstrap_ci(const ** 2, None, days, root=True), (2.0, 2.0))
    err = rng.normal(1, 1, (720, 5))
    lo, hi = metrics.day_bootstrap_ci(err, None, days)
    assert lo < err.mean() < hi
    assert metrics.day_bootstrap_ci(err, None, days) == (lo, hi)        # seeded


def test_ci_resamples_days_not_hours():
    # each day shares one error level: 720 hours but only 30 independent values, so the CI must be about as wide
    # as for 30 points, far wider than an hour-level bootstrap would give
    rng = np.random.default_rng(1)
    days = np.repeat(np.arange(30), 24)
    err = np.repeat(rng.normal(0, 1, 30), 24)[:, None]
    lo, hi = metrics.day_bootstrap_ci(err, None, days)
    assert hi - lo > 0.5                       # ~ 2 * 1.96 / sqrt(30) = 0.72; hour-iid would give ~0.15


def test_ci_mask_and_empty_days():
    days = np.repeat(np.arange(4), 2)
    vals = np.array([1., 1., 9., 9., 1., 1., 1., 1.])[:, None]
    mask = np.array([True, True, False, False, True, True, True, True])[:, None]   # day 1 has no real entry
    assert metrics.day_bootstrap_ci(vals, mask, days) == (1.0, 1.0)


def test_summarize_ci_keys():
    preds = np.random.default_rng(2).normal(size=(48, 3, 4))
    rows = metrics.summarize(preds, np.zeros_like(preds), np.ones_like(preds, bool), [0, 1, 2, 3],
                             days=np.repeat([0, 1], 24))
    lo, hi = rows[1]['t_rmse_ci']
    assert lo <= rows[1]['t_rmse'] <= hi
    assert '[' in metrics.to_markdown(rows)
    assert '_ci' not in str(metrics.summarize(preds, np.zeros_like(preds), np.ones_like(preds, bool), [0, 1, 2, 3])[1])


def test_target_days_uses_last_column():
    t = np.array([[0, 23 * 3600 * 10**6], [3600 * 10**6, 24 * 3600 * 10**6]])   # us, target 23:00 day 0 / 00:00 day 1
    d = metrics.target_days(t)
    assert d[0] != d[1]
