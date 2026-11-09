"""Score several runs on the same stations, hours and real entries, so their rows are directly comparable.

Each run of notebook 03 is otherwise scored on its own station set (358 or 26) and its own real hours (gap injection
removes some), which makes e.g. `sparse` look better than `base` for the wrong reason.

  stations  those of the reference run, matched by lon/lat in every other run
  hours     target hours present in every run
  entries   real in every run (so injected gaps of one run are dropped for all)

usage: python scripts/eval/common_set.py --ref <run dir> <run dir> [<run dir> ...]
"""

import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Source'))  # madis_network.pkl unpickles a Source class
import metrics

log = logging.getLogger('common_set')


def station_xy(run_dir):
    net = pickle.load(open(Path(run_dir) / 'madis_network.pkl', 'rb'))
    return np.c_[net.madis_lon.numpy().ravel(), net.madis_lat.numpy().ravel()]


def align(runs, ref_xy, tol=1e-4):
    """runs: {name: (xy, times, preds, targets, is_real)} with times the target-hour column of the Times pickle.

    Returns ({name: (preds, targets)} on the common stations/hours, joint real mask, common times).
    """
    common = np.array(sorted(set.intersection(*[set(r[1].tolist()) for r in runs.values()])))
    out, masks = {}, []
    for name, (xy, times, preds, targets, is_real) in runs.items():
        st = np.argmin(np.abs(xy[None] - ref_xy[:, None]).sum(-1), 1)
        if np.abs(xy[st] - ref_xy).max() > tol:
            raise ValueError(f'{name}: a reference station is missing')
        order = np.argsort(times)
        hr = order[np.searchsorted(times, common, sorter=order)]
        out[name] = (preds[hr][:, st], targets[hr][:, st])
        masks.append(is_real[hr][:, st])
    mask = np.logical_and.reduce(masks)
    first = next(iter(out.values()))[1]
    for name, (_, t) in out.items():
        if not np.allclose(t[mask], first[mask], atol=1e-3):
            raise ValueError(f'{name}: targets differ from the other runs on jointly real entries')
    return out, mask, common


def common_scores(run_dirs, ref_dir):
    """{run name: metrics.summarize real-only row with CIs}, all on the reference run's stations."""
    runs, order = {}, None
    for d in map(Path, run_dirs):
        preds, targets, is_real, order = metrics.load_run(d)
        times = pickle.load(open(d / 'Times_CUSTOM_min.pkl', 'rb'))[:, -1]
        runs[d.name] = (station_xy(d), times, preds, targets, is_real)
    out, mask, common = align(runs, station_xy(ref_dir))
    days = metrics.target_days(common[:, None])
    log.info('%d stations, %d hours, real fraction per variable %s', mask.shape[1], mask.shape[0],
             np.round(mask.mean((0, 1)), 3))
    return {n: metrics.summarize(p, t, mask, order, days=days)[1] for n, (p, t) in out.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--ref', required=True, type=Path, help='run whose stations are scored')
    parser.add_argument('run_dirs', nargs='+', type=Path)
    args = parser.parse_args(argv)
    try:
        rows = common_scores(args.run_dirs, args.ref)
    except FileNotFoundError as e:
        log.error('missing test output: %s', e)
        return 1
    for name, r in rows.items():
        r['entries'] = name
    sys.stdout.write(metrics.to_markdown(list(rows.values())).replace('| entries |', '| run |') + '\n')
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
