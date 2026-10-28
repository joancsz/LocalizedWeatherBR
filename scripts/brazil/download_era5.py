"""Download hourly ERA5 single-level fields, one NetCDF per month, from the Copernicus CDS.

usage: python scripts/brazil/download_era5.py --data-root $DATA_ROOT [--years 2019 ... 2023]
writes <data-root>/brazil/ERA5/<era5-region>/surface_{year}_{month}.nc in the format ERA5.load_ERA5_monthly reads.

Credentials: the first of $CDSAPI_RC, ~/.cdsapirc, <repo>/.cdsapirc, <data-root>/.cdsapirc (keep it on Drive for Colab).
"""

import argparse
import calendar
import logging
import os
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Source'))
import Utils.Paths as Paths  # noqa: E402

DATASET = 'reanalysis-era5-single-levels'
VARIABLES = ['10m_u_component_of_wind', '10m_v_component_of_wind', '2m_dewpoint_temperature', '2m_temperature',
             'surface_net_solar_radiation', 'surface_pressure']  # as Source/DataDownload/ERA5/DownloadERA5.ipynb
KEEP = ['u10', 'v10', 't2m', 'd2m', 'ssr', 'sp']
LOADER_BUFFER = 1.5  # ERA5.__init__ slices floor(low - 1.5) .. ceil(up + 1.5)

log = logging.getLogger('download_era5')


def find_cdsapirc(data_root):
    candidates = [os.environ.get('CDSAPI_RC'), Path.home() / '.cdsapirc',
                  Path(__file__).resolve().parents[2] / '.cdsapirc', data_root / '.cdsapirc']
    return next((Path(c) for c in candidates if c and Path(c).exists()), None)


def loader_box(shapefile):
    """[N, W, S, E] equal to the integer box ERA5.__init__ slices, so the request covers it exactly."""
    lon_low, lat_low, lon_up, lat_up = gpd.read_file(shapefile).bounds.iloc[0]
    return [float(np.ceil(lat_up + LOADER_BUFFER)), float(np.floor(lon_low - LOADER_BUFFER)),
            float(np.floor(lat_low - LOADER_BUFFER)), float(np.ceil(lon_up + LOADER_BUFFER))]


def check(path, year, month, area):
    """Post-check: full month of hours, no NaN, latitude descending (I5), area covered."""
    n, w, s, e = area
    with xr.open_dataset(path) as ds:
        hours = calendar.monthrange(year, month)[1] * 24
        assert ds.sizes['valid_time'] == hours, f'{ds.sizes["valid_time"]} hours, expected {hours}'
        assert all(v in ds for v in KEEP[:5]), f'missing variables, have {list(ds.data_vars)}'
        assert not any(bool(ds[v].isnull().any()) for v in KEEP[:5]), 'NaN in data'
        lat, lon = ds.latitude.values, ds.longitude.values
        assert (np.diff(lat) < 0).all(), 'latitude not descending'
        assert lat.max() >= n and lat.min() <= s and lon.min() <= w and lon.max() >= e, 'area not covered'


def merge_zip(part, dest):
    """New CDS returns a zip (instant + accum streams) when ssr is mixed with instantaneous fields."""
    folder = part.with_suffix('.unzipped')
    with zipfile.ZipFile(part) as z:
        z.extractall(folder)
    parts = [xr.open_dataset(f) for f in sorted(folder.glob('*.nc'))]
    try:
        xr.merge(parts, compat='override').to_netcdf(dest)
    finally:
        for p in parts:
            p.close()
    for f in folder.iterdir():
        f.unlink()
    folder.rmdir()


def fetch_month(client, out_dir, year, month, area):
    dest = Paths.era5_monthly_file(out_dir.parent.parent, out_dir.name, year, month)
    if dest.exists():
        try:
            check(dest, year, month, area)
            log.info('%s ok, skipping', dest.name)
            return
        except AssertionError as e:
            log.warning('%s failed its check (%s), downloading again', dest.name, e)
    part = dest.with_suffix('.part')
    request = {
        'product_type': ['reanalysis'], 'variable': VARIABLES, 'year': [str(year)], 'month': [f'{month:02d}'],
        'day': [f'{d:02d}' for d in range(1, calendar.monthrange(year, month)[1] + 1)],
        'time': [f'{h:02d}:00' for h in range(24)], 'data_format': 'netcdf', 'download_format': 'unarchived',
        'area': area,
    }
    log.info('requesting %s', dest.name)
    client.retrieve(DATASET, request).download(str(part))
    if zipfile.is_zipfile(part):
        merge_zip(part, dest)
        part.unlink()
    else:
        part.replace(dest)
    check(dest, year, month, area)
    log.info('wrote %s', dest)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--years', nargs='+', type=int, default=[2019, 2020, 2021, 2022, 2023])
    parser.add_argument('--months', nargs='+', type=int, default=list(range(1, 13)))
    parser.add_argument('--shapefile', default='Shapefiles/Regions/ne_brazil_buffered.shp',
                        help='relative to <data-root>/brazil')
    parser.add_argument('--era5-region', default='NEBrazil', help='era5_region in the config')
    parser.add_argument('--workers', type=int, default=4, help='parallel CDS requests')
    args = parser.parse_args(argv)

    data_path = args.data_root / 'brazil'
    shapefile = data_path / args.shapefile
    if not shapefile.exists():
        log.error('%s missing; run build_region.py first', shapefile)
        return 1
    rc = find_cdsapirc(args.data_root)
    if rc is None and not os.environ.get('CDSAPI_KEY'):
        log.error('no CDS credentials: put .cdsapirc in ~, the repo root or %s '
                  '(see https://cds.climate.copernicus.eu/how-to-api)', args.data_root)
        return 1
    if rc is not None:
        os.environ['CDSAPI_RC'] = str(rc)

    import cdsapi
    client = cdsapi.Client(quiet=True)
    area = loader_box(shapefile)
    out_dir = data_path / 'ERA5' / args.era5_region
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info('area [N, W, S, E] = %s', area)

    jobs = [(y, m) for y in args.years for m in args.months]
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(fetch_month, client, out_dir, y, m, area): (y, m) for y, m in jobs}
    failed = []
    for f, (y, m) in futures.items():
        try:
            f.result()
        except Exception as e:
            log.error('%d-%02d: %s', y, m, e)
            failed.append((y, m))
    if failed:
        log.error('%d of %d months failed; rerun to retry them', len(failed), len(jobs))
        return 1
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
