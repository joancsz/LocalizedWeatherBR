"""Quality check of single stations of a run: is a large error the model's or the observation's?

usage: python scripts/eval/station_check.py --run-dir <run> --stations 26 14

Station numbers are the run's 1-based order (as in diurnal/stations.csv). For each station prints the INMET code and
name, error stats of model / ERA5 / persistence on real entries, and observation red flags: hour-to-hour jumps, stuck
values, calm hours. Writes <run>/station_check/station_<n>.png: test-year T and wind speed, observed vs model vs ERA5.
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines
import metrics
from diurnal_analysis import COLORS, GRID, INK_2, MUTED, SURFACE

log = logging.getLogger('station_check')


def longest_run(x):
    """Length of the longest run of identical consecutive values."""
    if len(x) == 0:
        return 0
    change = np.flatnonzero(np.diff(x) != 0)
    edges = np.concatenate([[-1], change, [len(x) - 1]])
    return int(np.diff(edges).max())


def obs_flags(obs, real):
    """Red flags of one observed series (hourly, real mask): big jumps and stuck values over real hours only."""
    v = obs[real]
    jumps = np.abs(np.diff(obs))[real[1:] & real[:-1]]
    return dict(n_real=int(real.sum()), jumps_gt5=int((jumps > 5).sum()), max_jump=float(jumps.max(initial=0)),
                longest_stuck_h=longest_run(np.round(v, 2)))


def check(forecasts, targets, is_real, madis_vars_o, times, n):
    s = n - 1
    t = metrics.to_datetime(times[:, -1])
    out = {}
    for var, k in (('T', madis_vars_o.index(metrics.TEMP)), ('Td', madis_vars_o.index(metrics.DEWPOINT))):
        m = is_real[:, s, k]
        obs = targets[:, s, k]
        for name, f in forecasts.items():
            e = (f[:, s, k] - obs)[m]
            out[f'{var} {name}'] = dict(rmse=float(np.sqrt((e ** 2).mean())), bias=float(e.mean()),
                                        gt5=int((np.abs(e) > 5).sum()))
        # one target hour per sample; sort by it to get the hourly series
        order = np.argsort(t.values)
        out[f'{var} obs'] = obs_flags(obs[order], m[order])
    iu, iv = madis_vars_o.index(metrics.U), madis_vars_o.index(metrics.V)
    m = is_real[:, s, iu] & is_real[:, s, iv]
    speed = lambda a: np.hypot(a[:, s, iu], a[:, s, iv])[m]
    out['wind speed mean'] = {name: float(speed(f).mean()) for name, f in {'obs': targets, **forecasts}.items()}
    out['wind calm fraction (obs < 0.5 m/s)'] = float((speed(targets) < 0.5).mean())
    return out


def plot(forecasts, targets, is_real, madis_vars_o, times, n, title, path):
    s = n - 1
    t = metrics.to_datetime(times[:, -1])
    order = np.argsort(t.values)
    t = t[order]
    k, iu, iv = (madis_vars_o.index(v) for v in (metrics.TEMP, metrics.U, metrics.V))
    series = {'observed': targets, 'model': forecasts['model'], 'ERA5': forecasts['ERA5']}
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True, facecolor=SURFACE)
    for ax, (label, get, real) in zip(axes, (
            ('T (K)', lambda a: a[order, s, k], is_real[order, s, k]),
            ('wind speed (m/s)', lambda a: np.hypot(a[order, s, iu], a[order, s, iv]),
             is_real[order, s, iu] & is_real[order, s, iv]))):
        for name, a in series.items():
            y = np.where(real, get(a), np.nan)  # gaps where the observation is not real
            ax.plot(t, y, color=COLORS[name], linewidth=0.6 if name != 'observed' else 0.8, label=name,
                    alpha=0.9 if name == 'observed' else 0.7)
        ax.set_facecolor(SURFACE)
        ax.set_ylabel(label, color=INK_2)
        ax.grid(axis='y', color=GRID, linewidth=0.8)
        ax.tick_params(colors=MUTED)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    axes[0].set_title(title, loc='left', fontsize=11)
    axes[0].legend(frameon=False, fontsize=9, ncol=3, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--stations', required=True, type=int, nargs='+', help="1-based, the run's station order")
    args = parser.parse_args(argv)
    try:
        forecasts, targets, is_real, madis_vars_o, times, _ = baselines.load_forecasts(args.run_dir)
    except FileNotFoundError as e:
        log.error('missing input: %s', e)
        return 1
    _, meta = baselines.run_meta(args.run_dir)
    info = meta.stations.reset_index(drop=True)
    out = args.run_dir / 'station_check'
    out.mkdir(exist_ok=True)
    for n in args.stations:
        if not 1 <= n <= targets.shape[1]:
            log.error('station %d not in 1..%d', n, targets.shape[1])
            return 1
        row = info.iloc[n - 1]
        title = f'station {n}: {row.get("code", "?")} {row.get("name", "")} (elv {row.get("elv", float("nan")):.0f} m)'
        sys.stdout.write(f'\n== {title}\n')
        for key, val in check(forecasts, targets, is_real, madis_vars_o, times, n).items():
            fmt = (lambda v: f'{v:.3f}' if isinstance(v, float) else str(v))
            sys.stdout.write(f'  {key}: ' + (', '.join(f'{a}={fmt(b)}' for a, b in val.items())
                                             if isinstance(val, dict) else fmt(val)) + '\n')
        plot(forecasts, targets, is_real, madis_vars_o, times, n, title, out / f'station_{n}.png')
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
