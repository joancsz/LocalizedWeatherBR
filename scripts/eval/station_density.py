"""E4: score a run on a station set shared by every run of a sweep, so the numbers compare across runs.

  --eval-set fixed      density sweep (station_keep_fraction): the station_eval_n spread stations every subsample keeps
  --eval-set ratio:0.8  control-ratio sweep: the stations kept at ratio 0.8, present in every looser filter too

usage: python scripts/eval/station_density.py --run-dir <run> --eval-set fixed|ratio:<r>
Writes eval_<set>.csv in the run folder (model, ERA5, persistence; real observations only, day-bootstrap CIs) with the
run's station count, keep fraction, seed and control ratio, and prints a markdown table.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines
import metrics
from Utils.Degrade import eval_stations

log = logging.getLogger('station_density')


def match_stations(run_lons, run_lats, eval_lons, eval_lats):
    """Indices into the run's stations of each evaluation station (coordinates compared at 4 decimals, as in I1)."""
    key = lambda lo, la: list(zip(np.round(lo, 4), np.round(la, 4)))
    index = {k: i for i, k in enumerate(key(run_lons, run_lats))}
    missing = [k for k in key(eval_lons, eval_lats) if k not in index]
    if missing:
        raise ValueError(f'{len(missing)} evaluation stations are not in this run, e.g. {missing[0]}')
    return np.array([index[k] for k in key(eval_lons, eval_lats)])


def eval_coords(run_dir, eval_set):
    p, meta = baselines.run_meta(run_dir)
    if eval_set == 'fixed':
        # the run's ratio-filtered network before subsampling; same call as Degrade.keep_fraction
        base = meta.stations_source if p.get('station_keep_fraction') is not None else meta.stations
        g = base.geometry.iloc[eval_stations(base.geometry.x.values, base.geometry.y.values, p.get('station_eval_n', 6))]
    else:
        ratio = float(eval_set.split(':')[1])
        _, ref = baselines.run_meta(run_dir, madis_control_ratio=ratio, station_keep_n=None, station_keep_fraction=None)
        g = ref.stations.geometry
    return p, g.x.values, g.y.values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--eval-set', required=True, help="'fixed' or 'ratio:<r>'")
    args = parser.parse_args(argv)
    try:
        forecasts, targets, is_real, madis_vars_o, times, madis = baselines.load_forecasts(args.run_dir)
        p, lons, lats = eval_coords(args.run_dir, args.eval_set)
    except FileNotFoundError as e:
        log.error('missing input: %s', e)
        return 1
    sel = match_stations(madis.lon.values, madis.lat.values, lons, lats)
    log.info('%s: scoring %d of %d stations', args.eval_set, len(sel), targets.shape[1])
    rows = baselines.score({k: v[:, sel] for k, v in forecasts.items()}, targets[:, sel], is_real[:, sel],
                           madis_vars_o, times)
    baselines.write_rows(rows, args.run_dir / f'eval_{args.eval_set.replace(":", "")}.csv',
                         run=args.run_dir.name, n_stations=targets.shape[1], n_eval=len(sel),
                         keep_fraction=p.get('station_keep_fraction') or 1.0, seed=p.get('station_subsample_seed', 0),
                         control_ratio=p['madis_control_ratio'])
    sys.stdout.write(metrics.to_markdown(rows).replace('| entries |', '| forecast |') + '\n')
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
