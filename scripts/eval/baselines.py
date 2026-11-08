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


def to_datetime(times):
    # Times pickles hold int64 epoch time; pandas may write ns or us depending on version
    return pd.to_datetime(times, unit='us' if np.abs(times).max() < 1e17 else 'ns')


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


def load_inputs(run_dir, years):
    p = json.load(open(run_dir / 'params.json'))
    data_path = Path(p['data_path'])
    if p['shapefile_path'] is None:
        lon_low, lon_up, lat_low, lat_up = p['coords']
        shapefile = None
    else:
        shapefile = data_path / p['shapefile_path']
        lon_low, lat_low, lon_up, lat_up = gpd.read_file(shapefile).bounds.iloc[0].values
    # same constructor call as Main.Run, so file names (including station_keep_n) match the run's
    meta = MetaStation(lat_low, lat_up, lon_low, lon_up, p['n_years'], p['madis_control_ratio'],
                       shapefile_path=shapefile, data_path=data_path, start_year=p['start_year'],
                       end_year=p['end_year'], station_keep_n=p.get('station_keep_n'),
                       station_subsample_seed=p.get('station_subsample_seed', 0))
    tag = Paths.era5_grid_tag(p.get('era5_coarsen_deg', 0.25))
    madis = open_years([Paths.madis_processed_file(data_path, meta.start_year, meta.end_year, y,
                                                   meta.filtered_file_name) for y in years])
    era5 = open_years([Paths.era5_interpolated_file(data_path, 1, 'Nearest', y, meta.filtered_file_name, tag)
                       for y in years])
    return madis, era5, p['lead_hrs']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        preds, targets, is_real, madis_vars_o = metrics.load_run(args.run_dir)
        times = pickle.load(open(args.run_dir / 'Times_CUSTOM_min.pkl', 'rb'))
        years = sorted(set(to_datetime(times[:, -1]).year))
        madis, era5, lead_hrs = load_inputs(args.run_dir, years)
    except FileNotFoundError as e:
        log.error('missing input: %s', e)
        return 1
    assert madis.sizes['stations'] == era5.sizes['stations'] == preds.shape[1], 'station count mismatch with the run'

    rows = []
    for name, p in {'model': preds, **baseline_preds(times, lead_hrs, madis, era5, madis_vars_o)}.items():
        row = metrics.summarize(p, targets, is_real, madis_vars_o)[1]  # real observations only
        row['entries'] = name
        rows.append(row)
    pd.DataFrame(rows).to_csv(args.run_dir / 'baselines.csv', index=False)
    sys.stdout.write(metrics.to_markdown(rows).replace('| entries |', '| forecast |') + '\n')
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
