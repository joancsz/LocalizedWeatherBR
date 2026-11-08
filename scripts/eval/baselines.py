"""Reference forecasts on the same test samples as a trained run, scored on real observations only.

  ERA5        ERA5 nearest grid point at the target time t+l (the Nearest interpolated file)
  persistence w(t+l) = w(t), the station's last value at issue time (filled hours carry the last observation)

usage: python scripts/eval/baselines.py --run-dir <data_path>/<output_saving_path>
Reads the run's params.json and test pickles; writes baselines.csv next to them and prints a markdown table.
The ERA5 baseline needs the Nearest interpolated file of the run's stations, which any Transformer + ERA5 Nearest
run builds (for a GNN run, run the Transformer with the same stations first).
"""

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Source'))
import metrics
import Utils.Paths as Paths
from Dataloader.MetaStation import MetaStation

NAMES = ['u', 'v', 'temp', 'dewpoint']  # EnvVariables order, same names in the processed and interpolated files

log = logging.getLogger('baselines')


to_datetime = metrics.to_datetime


def baseline_preds(times, lead_hrs, madis_ds, era5_ds, madis_vars_o):
    """{'ERA5', 'persistence'} -> (samples, stations, len(madis_vars_o)) arrays aligned with the run's Preds.

    times: (samples, back_hrs + lead_hrs + 1) sample time axis; the last column is the target hour.
    madis_ds / era5_ds: (stations, time) datasets in the run's station order, covering the test years.
    """
    target = to_datetime(times[:, -1])
    issue = to_datetime(times[:, -1 - lead_hrs])
    names = [NAMES[k] for k in madis_vars_o]
    pick = lambda ds, t: np.stack([ds[n].sel(time=t).transpose('time', 'stations').values for n in names], -1)
    return {'ERA5': pick(era5_ds, target), 'persistence': pick(madis_ds, issue)}


def open_years(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(', '.join(missing))
    return xr.concat([xr.open_dataset(p).load() for p in paths], dim='time', data_vars='minimal')


def run_meta(run_dir, **override):
    """(params, MetaStation) of a run, built with the same constructor call as Main.Run so file names match.

    override replaces params keys, e.g. madis_control_ratio=0.8 for another station set of the same region.
    """
    p = {**json.load(open(run_dir / 'params.json')), **override}
    data_path = Path(p['data_path'])
    if p['shapefile_path'] is None:
        lon_low, lon_up, lat_low, lat_up = p['coords']
        shapefile = None
    else:
        shapefile = data_path / p['shapefile_path']
        lon_low, lat_low, lon_up, lat_up = gpd.read_file(shapefile).bounds.iloc[0].values
    meta = MetaStation(lat_low, lat_up, lon_low, lon_up, p['n_years'], p['madis_control_ratio'],
                       shapefile_path=shapefile, data_path=data_path, start_year=p['start_year'],
                       end_year=p['end_year'], station_keep_n=p.get('station_keep_n'),
                       station_subsample_seed=p.get('station_subsample_seed', 0),
                       station_keep_fraction=p.get('station_keep_fraction'),
                       station_eval_n=p.get('station_eval_n', 6))
    return p, meta


def load_inputs(run_dir, years):
    p, meta = run_meta(run_dir)
    data_path = Path(p['data_path'])
    tag = Paths.era5_grid_tag(p.get('era5_coarsen_deg', 0.25))
    madis = open_years([Paths.madis_processed_file(data_path, meta.start_year, meta.end_year, y,
                                                   meta.filtered_file_name) for y in years])
    era5 = open_years([Paths.era5_interpolated_file(data_path, 1, 'Nearest', y, meta.filtered_file_name, tag)
                       for y in years])
    return madis, era5, p['lead_hrs']


def load_forecasts(run_dir):
    """Everything the evaluation scripts share: forecasts {'model', 'ERA5', 'persistence'} -> (samples, stations, vars),
    plus targets, is_real, madis_vars_o, the Times pickle and the processed station dataset (for lon/lat)."""
    preds, targets, is_real, madis_vars_o = metrics.load_run(run_dir)
    times = pickle.load(open(run_dir / 'Times_CUSTOM_min.pkl', 'rb'))
    years = sorted(set(to_datetime(times[:, -1]).year))
    madis, era5, lead_hrs = load_inputs(run_dir, years)
    assert madis.sizes['stations'] == era5.sizes['stations'] == preds.shape[1], 'station count mismatch with the run'
    forecasts = {'model': preds, **baseline_preds(times, lead_hrs, madis, era5, madis_vars_o)}
    return forecasts, targets, is_real, madis_vars_o, times, madis


def score(forecasts, targets, is_real, madis_vars_o, times):
    """Real-only metric row with day-bootstrap CIs per forecast."""
    days = metrics.target_days(times)
    rows = []
    for name, p in forecasts.items():
        row = metrics.summarize(p, targets, is_real, madis_vars_o, days=days)[1]
        row['entries'] = name
        rows.append(row)
    return rows


def write_rows(rows, path, **extra):
    # CI tuples become <metric>_lo / <metric>_hi columns; extra adds constant columns (e.g. run settings)
    flat = [{**extra, **{k2: v2 for k, v in r.items() for k2, v2 in
                         (zip((k[:-3] + '_lo', k[:-3] + '_hi'), v) if k.endswith('_ci') else [(k, v)])}} for r in rows]
    pd.DataFrame(flat).to_csv(path, index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        forecasts, targets, is_real, madis_vars_o, times, _ = load_forecasts(args.run_dir)
    except FileNotFoundError as e:
        log.error('missing input: %s', e)
        return 1
    rows = score(forecasts, targets, is_real, madis_vars_o, times)
    write_rows(rows, args.run_dir / 'baselines.csv')
    sys.stdout.write(metrics.to_markdown(rows).replace('| entries |', '| forecast |') + '\n')
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
