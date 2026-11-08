"""Test metrics in physical units from the pickles written by Source/Main.SaveTest.

usage: python scripts/eval/metrics.py --run-dir <data_path>/<output_saving_path>
Each metric is followed by its 95% confidence interval from a block bootstrap over days (1000 resamples).
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# EnvVariables values (Source/Settings/Settings.py)
U, V, TEMP, DEWPOINT = 0, 1, 2, 3

log = logging.getLogger('metrics')


def _sel(x, mask):
    return x if mask is None else x[mask]


def wind_vector_error(pred_u, pred_v, u, v, mask=None):
    return float(np.mean(_sel(np.hypot(pred_u - u, pred_v - v), mask)))


def rmse(pred, target, mask=None):
    return float(np.sqrt(np.mean(_sel((pred - target) ** 2, mask))))


def mae(pred, target, mask=None):
    return float(np.mean(_sel(np.abs(pred - target), mask)))


def bias(pred, target, mask=None):
    return float(np.mean(_sel(pred - target, mask)))


def day_bootstrap_ci(values, mask, days, root=False, n_boot=1000, seed=0):
    """95% CI of mean(values[mask]) (its sqrt if root, for RMSE) resampling whole days with replacement.

    values, mask: (samples, ...) per-entry errors and real flags; days: (samples,) day label of each sample.
    Hours within a day are correlated, so days, not hours, are the independent units.
    """
    codes = pd.factorize(np.asarray(days))[0]
    flat = lambda a: np.broadcast_to(a, values.shape).reshape(len(codes), -1)
    m = flat(np.ones(values.shape, bool) if mask is None else mask)
    sums = np.bincount(codes, np.where(m, flat(values), 0).sum(1))
    counts = np.bincount(codes, m.sum(1))
    sums, counts = sums[counts > 0], counts[counts > 0]  # a day with no real entry carries no information
    idx = np.random.default_rng(seed).integers(0, len(sums), (n_boot, len(sums)))
    est = sums[idx].sum(1) / counts[idx].sum(1)
    lo, hi = np.percentile(np.sqrt(est) if root else est, [2.5, 97.5])
    return float(lo), float(hi)


def to_datetime(times):
    # Times pickles hold int64 epoch time; pandas may write ns or us depending on version
    return pd.to_datetime(times, unit='us' if np.abs(times).max() < 1e17 else 'ns')


def target_days(times):
    """UTC day of each sample's target hour (last column of the Times pickle)."""
    return to_datetime(times[:, -1]).normalize().values


def load_run(run_dir):
    load = lambda name: pickle.load(open(run_dir / f'{name}_CUSTOM_min.pkl', 'rb'))
    params = run_dir / 'params.json'
    madis_vars_o = json.load(open(params))['madis_vars_o'] if params.exists() else [U, V, TEMP, DEWPOINT]
    return load('Preds'), load('Targets'), load('IsReal'), list(madis_vars_o)


def summarize(preds, targets, is_real, madis_vars_o, days=None):
    """Metric rows for all entries and real only; with days (see target_days) each metric also gets a '<name>_ci'."""
    k = {var: madis_vars_o.index(var) for var in (U, V, TEMP, DEWPOINT)}
    rows = []
    for label, use_mask in (('all', False), ('real only', True)):
        m = lambda *vars: np.logical_and.reduce([is_real[..., k[v]] for v in vars]) if use_mask else None
        p, t = lambda v: preds[..., k[v]], lambda v: targets[..., k[v]]
        row = dict(
            entries=label,
            wind=wind_vector_error(p(U), p(V), t(U), t(V), m(U, V)),
            t_rmse=rmse(p(TEMP), t(TEMP), m(TEMP)),
            td_rmse=rmse(p(DEWPOINT), t(DEWPOINT), m(DEWPOINT)),
            t_mae=mae(p(TEMP), t(TEMP), m(TEMP)),
            td_mae=mae(p(DEWPOINT), t(DEWPOINT), m(DEWPOINT)),
            t_bias=bias(p(TEMP), t(TEMP), m(TEMP)),
            td_bias=bias(p(DEWPOINT), t(DEWPOINT), m(DEWPOINT)),
        )
        if days is not None:
            err = lambda v: p(v) - t(v)
            # (per-entry values, mask, root) whose masked mean (sqrt for RMSE) is the metric
            per_entry = dict(
                wind=(np.hypot(err(U), err(V)), m(U, V), False),
                t_rmse=(err(TEMP) ** 2, m(TEMP), True),
                td_rmse=(err(DEWPOINT) ** 2, m(DEWPOINT), True),
                t_mae=(np.abs(err(TEMP)), m(TEMP), False),
                td_mae=(np.abs(err(DEWPOINT)), m(DEWPOINT), False),
                t_bias=(err(TEMP), m(TEMP), False),
                td_bias=(err(DEWPOINT), m(DEWPOINT), False),
            )
            for name, (values, mask, root) in per_entry.items():
                row[f'{name}_ci'] = day_bootstrap_ci(values, mask, days, root)
        rows.append(row)
    return rows


def to_markdown(rows):
    lines = ['| entries | wind-vector error (m/s) | T RMSE (K) | Td RMSE (K) | T MAE | Td MAE | T bias | Td bias |',
             '|---|---|---|---|---|---|---|---|']
    names = ['wind', 't_rmse', 'td_rmse', 't_mae', 'td_mae', 't_bias', 'td_bias']
    cell = lambda r, n: f'{r[n]:.3f}' + (' [{:.3f}, {:.3f}]'.format(*r[f'{n}_ci']) if f'{n}_ci' in r else '')
    for r in rows:
        lines.append(f'| {r["entries"]} | ' + ' | '.join(cell(r, n) for n in names) + ' |')
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        days = target_days(pickle.load(open(args.run_dir / 'Times_CUSTOM_min.pkl', 'rb')))
        rows = summarize(*load_run(args.run_dir), days=days)
    except FileNotFoundError as e:
        log.error('missing test output: %s', e)
        return 1
    sys.stdout.write(to_markdown(rows) + '\n')
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
