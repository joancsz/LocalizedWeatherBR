"""Test metrics in physical units from the pickles written by Source/Main.SaveTest.

usage: python scripts/eval/metrics.py --run-dir <data_path>/<output_saving_path>
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

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


def load_run(run_dir):
    load = lambda name: pickle.load(open(run_dir / f'{name}_CUSTOM_min.pkl', 'rb'))
    params = run_dir / 'params.json'
    madis_vars_o = json.load(open(params))['madis_vars_o'] if params.exists() else [U, V, TEMP, DEWPOINT]
    return load('Preds'), load('Targets'), load('IsReal'), list(madis_vars_o)


def summarize(preds, targets, is_real, madis_vars_o):
    k = {var: madis_vars_o.index(var) for var in (U, V, TEMP, DEWPOINT)}
    rows = []
    for label, use_mask in (('all', False), ('real only', True)):
        m = lambda *vars: np.logical_and.reduce([is_real[..., k[v]] for v in vars]) if use_mask else None
        p, t = lambda v: preds[..., k[v]], lambda v: targets[..., k[v]]
        rows.append(dict(
            entries=label,
            wind=wind_vector_error(p(U), p(V), t(U), t(V), m(U, V)),
            t_rmse=rmse(p(TEMP), t(TEMP), m(TEMP)),
            td_rmse=rmse(p(DEWPOINT), t(DEWPOINT), m(DEWPOINT)),
            t_mae=mae(p(TEMP), t(TEMP), m(TEMP)),
            td_mae=mae(p(DEWPOINT), t(DEWPOINT), m(DEWPOINT)),
            t_bias=bias(p(TEMP), t(TEMP), m(TEMP)),
            td_bias=bias(p(DEWPOINT), t(DEWPOINT), m(DEWPOINT)),
        ))
    return rows


def to_markdown(rows):
    lines = ['| entries | wind-vector error (m/s) | T RMSE (K) | Td RMSE (K) | T MAE | Td MAE | T bias | Td bias |',
             '|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| {entries} | {wind:.3f} | {t_rmse:.3f} | {td_rmse:.3f} | {t_mae:.3f} | {td_mae:.3f} | '
                     '{t_bias:.3f} | {td_bias:.3f} |'.format(**r))
    return '\n'.join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        rows = summarize(*load_run(args.run_dir))
    except FileNotFoundError as e:
        log.error('missing test output: %s', e)
        return 1
    sys.stdout.write(to_markdown(rows) + '\n')
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
