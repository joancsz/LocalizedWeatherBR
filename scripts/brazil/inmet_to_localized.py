"""Turn INMET raw CSVs into the station shapefiles and processed NetCDFs that Source/ reads as MADIS.

usage: python scripts/brazil/inmet_to_localized.py --data-root $DATA_ROOT [--years 2019 ... 2023] [--control-ratio 0.6]

Reads  <data-root>/brazil/inmet/raw/{year}/**/*.CSV and the region shapefile from build_region.py.
Writes, under data_path = <data-root>/brazil (names from Source/Utils/Paths.py):
  madis/stations/stations_{start}_{end}_{file_name}.shp            all stations in the region (num, code, name)
  madis/processed/Meta-{start}-{end}/madis_{year}_{file_name}.nc   all stations, in that shapefile's row order
  the filtered shapefile and madis_{year}_{filtered_file_name}.nc  via MetaStation and Madis.createFile's selection
  inmet/station_counts.csv                                          retained stations per control ratio
Training with another madis_control_ratio needs no rerun: MetaStation and Madis derive it from the all-station files.
"""

import argparse
import logging
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Source'))
import Utils.Paths as Paths  # noqa: E402
from Dataloader.MetaStation import MetaStation  # noqa: E402

import inmet_parser as P  # noqa: E402

VARS = ['u', 'v', 'temp', 'dewpoint', 'solar_radiation']
RATIOS = [0.6, 0.7, 0.8, 0.9]
MIN_STATIONS = 25

log = logging.getLogger('inmet_to_localized')


def collect(raw_root, years, region):
    """{code: {'meta': most recent meta, 'frames': [df per file]}} for stations whose point is inside region."""
    stations = {}
    for year in years:
        files = sorted((raw_root / str(year)).rglob('*.[Cc][Ss][Vv]'))
        if not files:
            raise FileNotFoundError(f'no CSV files under {raw_root / str(year)}; run download_inmet.py first')
        n_in = 0
        for f in files:
            meta = P.read_meta(f)
            if np.isnan(meta['lon']) or np.isnan(meta['lat']):
                log.warning('%s: no coordinates, skipped', f.name)
                continue
            if not Point(round(meta['lon'], 4), round(meta['lat'], 4)).within(region):
                continue
            meta, header, df = P.parse_file(f)
            if n_in == 0:
                log.info('%d header (%s): %s', year, f.name, header)
            n_in += 1
            entry = stations.setdefault(meta['code'], {'meta': meta, 'frames': []})
            old = entry['meta']
            if max(abs(old['lon'] - meta['lon']), abs(old['lat'] - meta['lat'])) > 0.01:
                log.warning('%s moved from (%.4f, %.4f) to (%.4f, %.4f) in %d; using the newer position',
                            meta['code'], old['lon'], old['lat'], meta['lon'], meta['lat'], year)
            entry['meta'] = meta  # years ascend, so the last one wins
            entry['frames'].append(df)
        log.info('%d: %d of %d files inside the region', year, n_in, len(files))
    return stations


def assemble(stations, timeline, region):
    """Per-station QC + conversion on the full hourly timeline. Returns (GeoDataFrame, {code: converted df})."""
    rows, series, temps = [], {}, {}
    for code in sorted(stations):
        meta = stations[code]['meta']
        point = Point(round(meta['lon'], 4), round(meta['lat'], 4))  # I1: same rounding in shapefile and NetCDF
        if not point.within(region):
            log.warning('%s: latest position outside the region, dropped', code)
            continue
        df = pd.concat(stations[code]['frames']).sort_index()
        df = df[~df.index.duplicated()].reindex(timeline)
        clean, rejected = P.qc(df)
        log.info('%s QC rejected fraction: %s', code, ' '.join(f'{k}={v:.3f}' for k, v in rejected.items()))
        conv = P.to_localized(clean)
        temps[code] = clean.temp
        series[code] = conv
        num = int(conv[['u', 'v', 'temp']].notna().all(axis=1).sum())  # MADIS validobs: ws, wd and temp
        rows.append({'num': num, 'code': code, 'name': meta.get('name', ''), 'elv': meta['elv'], 'geometry': point})

    peaks = P.diurnal_peak_hours(temps)
    log.info('diurnal temperature peak hour (UTC) across stations: %s', peaks.value_counts().sort_index().to_dict())
    log.info('median peak %.1f UTC', P.check_timezone(peaks))
    return gpd.GeoDataFrame(rows, crs='EPSG:4326'), series


def year_dataset(gdf, series, year):
    """One year in the Madis.createRawFile schema and fill rule."""
    time = pd.date_range(f'{year}-01-01', f'{year}-12-31 23:00', freq='h')  # I2
    data = {}
    for var in VARS:
        values = np.stack([series[c][var].loc[time].to_numpy() for c in gdf.code])  # (stations, time)
        data[f'{var}_is_real'] = (['stations', 'time'], ~np.isnan(values))
        filled = pd.DataFrame(values.T).ffill().bfill().values.T
        if np.isnan(filled).all():
            raise ValueError(f'{year}: {var} has no observations at any station')
        data[var] = (['stations', 'time'], np.nan_to_num(filled, nan=np.nanmean(filled)))
    data['elv'] = (['stations'], gdf.elv.to_numpy(dtype=float))
    data['lon'] = (['stations'], gdf.geometry.x.to_numpy())
    data['lat'] = (['stations'], gdf.geometry.y.to_numpy())
    return xr.Dataset(data, coords={'stations': np.arange(1, len(gdf) + 1), 'time': time})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--years', nargs='+', type=int, default=[2019, 2020, 2021, 2022, 2023])
    parser.add_argument('--shapefile', default='Shapefiles/Regions/ne_brazil_buffered.shp',
                        help='relative to <data-root>/brazil, as shapefile_path in the config')
    parser.add_argument('--control-ratio', type=float, default=0.6)  # 0.8 keeps only 15 stations over 2019-2023
    args = parser.parse_args(argv)

    years = sorted(args.years)
    start, end = years[0], years[-1]
    if years != list(range(start, end + 1)):
        log.error('--years must be consecutive, got %s', years)
        return 1
    data_path = args.data_root / 'brazil'
    shapefile = data_path / args.shapefile
    if not shapefile.exists():
        log.error('%s missing; run build_region.py first', shapefile)
        return 1

    # same bounds and names as Main / MetaStation
    lon_low, lat_low, lon_up, lat_up = gpd.read_file(shapefile).bounds.iloc[0]
    region = gpd.read_file(shapefile).dissolve().geometry.iloc[0]
    file_name = Paths.region_file_name(lon_low, lon_up, lat_low, lat_up, shapefile)
    all_station_file = Paths.station_file(data_path, start, end, file_name)
    raw_files = {y: Paths.madis_processed_file(data_path, start, end, y, file_name) for y in years}

    try:
        if all_station_file.exists() and all(f.exists() for f in raw_files.values()):
            log.info('all-station shapefile and processed files exist, skipping parse')
            gdf = gpd.read_file(all_station_file)
        else:
            timeline = pd.date_range(f'{start}-01-01', f'{end}-12-31 23:00', freq='h')
            stations = collect(data_path / 'inmet' / 'raw', years, region)
            gdf, series = assemble(stations, timeline, region)
            if gdf.empty:
                raise ValueError('no INMET station inside the region')
            all_station_file.parent.mkdir(parents=True, exist_ok=True)
            Paths.madis_processed_dir(data_path, start, end).mkdir(parents=True, exist_ok=True)
            for y in years:
                ds = year_dataset(gdf, series, y)
                assert not any(np.isnan(ds[v]).any() for v in VARS), 'I4: NaN after fill'
                ds.to_netcdf(raw_files[y])
                log.info('wrote %s (%d stations)', raw_files[y], len(gdf))
            gdf.to_file(all_station_file)  # last, so its presence marks a finished run
            log.info('wrote %s', all_station_file)

        n_hours = len(pd.date_range(f'{start}-01-01', f'{end}-12-31 23:00', freq='h'))
        counts = pd.DataFrame({'control_ratio': RATIOS,
                               'n_stations': [int((gdf.num >= n_hours * r).sum()) for r in RATIOS]})
        counts.to_csv(data_path / 'inmet' / 'station_counts.csv', index=False)
        log.info('stations retained per control ratio:\n%s', counts.to_string(index=False))

        meta = MetaStation(lat_low, lat_up, lon_low, lon_up, control_ratio=args.control_ratio,
                           shapefile_path=shapefile, data_path=data_path, start_year=start, end_year=end)
        n = len(meta.stations)
        (log.warning if n < MIN_STATIONS else log.info)('%d stations at control ratio %s', n, args.control_ratio)
        if n == 0:
            raise ValueError(f'no station passes control ratio {args.control_ratio}')

        # Madis.createFile's selection, so the order matches the filtered shapefile (I1)
        coords_raw, coords = list(meta.stations_raw.geometry), list(meta.stations.geometry)
        ind = [e in coords for e in coords_raw]
        for y in years:
            out = Paths.madis_processed_file(data_path, start, end, y, meta.filtered_file_name)
            if out.exists():
                continue
            with xr.open_dataset(raw_files[y]) as raw:
                ds = raw.sel(stations=ind).load()
            ds['stations'] = np.arange(1, len(coords) + 1)
            ds.to_netcdf(out)
            log.info('wrote %s', out)
    except Exception as e:
        log.error('%s', e)
        return 1
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
