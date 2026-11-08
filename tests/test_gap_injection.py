"""PLAN_UPDATE_v2 §2.5 (2): synthetic station outages on top of the processed file.

Interface pinned here (to implement in Source/Utils/Degrade.py):
  Degrade.inject_gaps(is_real, target_real_fraction, mean_gap_hours, seed) -> bool array, same shape (stations, time)
      new mask is a subset of is_real; outages are contiguous blocks of hours
  Degrade.refill(values, is_real) -> values with non-real hours re-filled exactly like Madis.createRawFile
      (ffill then bfill along time, per station)
"""
import numpy as np
import pandas as pd
import pytest

Degrade = pytest.importorskip('Utils.Degrade', reason='gap injection not implemented yet (PLAN_UPDATE_v2 §2.5)')

N_ST, N_T = 30, 8760


def existing_mask(real_fraction=0.964, seed=0):
    # NE-US u/v/T real fraction on the test year is 0.964 (docs/repro_log.md P1)
    return np.random.default_rng(seed).random((N_ST, N_T)) < real_fraction


def runs_of_false(row):
    # lengths of consecutive False runs in a 1-D bool array
    d = np.diff(np.concatenate([[1], row.astype(int), [1]]))
    return np.flatnonzero(d == 1) - np.flatnonzero(d == -1)


@pytest.mark.parametrize('target', [0.9, 0.75, 0.6])
def test_final_real_fraction_within_one_percent(target):
    out = Degrade.inject_gaps(existing_mask(), target, mean_gap_hours=72, seed=0)
    assert out.shape == (N_ST, N_T) and out.dtype == bool
    assert abs(out.mean() - target) <= 0.01


def test_only_removes_never_adds():
    m = existing_mask()
    out = Degrade.inject_gaps(m, 0.8, mean_gap_hours=72, seed=0)
    assert not np.any(out & ~m)


def test_gaps_are_contiguous_blocks():
    m = np.ones((N_ST, N_T), dtype=bool)       # no pre-existing gaps, so every False is injected
    out = Degrade.inject_gaps(m, 0.8, mean_gap_hours=72, seed=0)
    runs = np.concatenate([runs_of_false(r) for r in out])
    assert runs.size > 0
    assert np.median(runs) >= 24                # outages, not scattered single hours
    assert 0.5 * 72 <= runs.mean() <= 2 * 72


def test_reproducible_by_seed():
    m = existing_mask()
    np.testing.assert_array_equal(Degrade.inject_gaps(m, 0.8, 72, seed=1), Degrade.inject_gaps(m, 0.8, 72, seed=1))
    assert not np.array_equal(Degrade.inject_gaps(m, 0.8, 72, seed=1), Degrade.inject_gaps(m, 0.8, 72, seed=2))


def test_input_not_modified():
    m = existing_mask()
    before = m.copy()
    Degrade.inject_gaps(m, 0.8, 72, seed=0)
    np.testing.assert_array_equal(m, before)


def test_target_above_current_is_noop():
    m = existing_mask(0.7)
    np.testing.assert_array_equal(Degrade.inject_gaps(m, 0.9, 72, seed=0), m)


def test_refill_matches_madis_rule():
    values = np.array([[1., 2., 3., 4., 5.], [10., 20., 30., 40., 50.]])
    is_real = np.array([[False, True, False, False, True], [True, True, True, True, True]])
    out = Degrade.refill(values, is_real)
    expected = pd.DataFrame(np.where(is_real, values, np.nan).T).ffill().bfill().values.T
    np.testing.assert_array_equal(out, expected)
    np.testing.assert_array_equal(out[0], [2., 2., 2., 2., 5.])
    assert not np.isnan(out).any()             # I4
