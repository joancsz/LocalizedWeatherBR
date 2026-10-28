"""Validate the NE Brazil dataset (after inmet_to_localized.py and download_era5.py) and write a report.

usage: python scripts/brazil/validate_dataset.py --data-root $DATA_ROOT [--years 2019 ... 2023] [--control-ratio 0.6]
writes <out>/brazil_data_report.md plus stations_map.png and availability.png (default out: <data-root>/brazil/report,
so it survives a Colab session; copy it to docs/). Exits non-zero if any check fails; the report is written either way.

Checks (PLAN_BRAZIL §1 invariants):
  I1 station order of every filtered NetCDF equals the filtered shapefile rows
  I2 hourly time axis of the full calendar year
  I3 ERA5 monthly files cover the same UTC hours, and the diurnal temperature peak at every station agrees with
     ERA5 at its nearest grid point within ±1 h
  I4 no NaN in the filled station variables
  I5 ERA5 latitude descending, longitude in [-180, 180]
"""

import argparse
import calendar
import logging
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import xarray as xr

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'Source'))
import Utils.Paths as Paths  # noqa: E402

FILLED = ['u', 'v', 'temp', 'dewpoint', 'solar_radiation']
PAIRS = {'u': 'u10', 'v': 'v10', 'temp': 't2m', 'dewpoint': 'd2m'}  # INMET name -> ERA5 name
RATIOS = [0.6, 0.7, 0.8, 0.9]
MIN_STATIONS = 25
SEASON_DAYS = ['01-15', '04-15', '07-15', '10-15']  # one date per season, in the last year
# reference palette (dataviz skill): sequential blue ramp, categorical slot 1, secondary ink
BLUE_RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
SERIES_1, INK_2, INK_1, SURFACE = '#2a78d6', '#52514e', '#0b0b0b', '#fcfcfb'

log = logging.getLogger('validate_dataset')


class Checks:
    def __init__(self):
        self.rows = []

    def __call__(self, name, ok, detail=''):
        self.rows.append((name, bool(ok), detail))
        (log.info if ok else log.error)('%s %s %s', 'PASS' if ok else 'FAIL', name, detail)

    @property
    def failed(self):
        return [r for r in self.rows if not r[1]]


def hour_gap(a, b):
    d = np.abs(np.asarray(a) - np.asarray(b)) % 24
    return np.minimum(d, 24 - d)


def era5_at_stations(era5_dir, years, lons, lats, check):
    """ERA5 at each station's nearest grid point, (stations, time); checks I3 coverage and I5 on every monthly file."""
    sel = dict(longitude=xr.DataArray(lons, dims='stations'), latitude=xr.DataArray(lats, dims='stations'))
    parts, bad_time, bad_grid = [], [], []
    for y in years:
        for m in range(1, 13):
            f = Paths.era5_monthly_file(era5_dir.parent.parent, era5_dir.name, y, m)
            if not f.exists():
                bad_time.append(f'{y}-{m:02d} missing')
                continue
            with xr.open_dataset(f) as ds:
                expected = pd.date_range(f'{y}-{m:02d}-01', periods=calendar.monthrange(y, m)[1] * 24, freq='h')
                if not pd.DatetimeIndex(ds.valid_time.values).equals(expected):
                    bad_time.append(f'{y}-{m:02d}')
                lat, lon = ds.latitude.values, ds.longitude.values
                if not ((np.diff(lat) < 0).all() and lon.min() >= -180 and lon.max() <= 180):
                    bad_grid.append(f'{y}-{m:02d}')
                parts.append(ds[list(PAIRS.values())].sel(**sel, method='nearest').load())
    check('I3 ERA5 hourly UTC coverage', not bad_time, ', '.join(bad_time) or f'{len(years) * 12} monthly files')
    check('I5 ERA5 lat descending, lon in [-180, 180]', not bad_grid, ', '.join(bad_grid))
    if not parts:
        return None
    return xr.concat(parts, dim='valid_time').rename(valid_time='time')


def peak_hour(da):
    """UTC hour of the mean diurnal maximum, per station."""
    return da.groupby('time.hour').mean().argmax('hour').values


def stations_map(path, region, stations, retained, ratio):
    fig, ax = plt.subplots(figsize=(7, 6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    gpd.GeoSeries([region]).boundary.plot(ax=ax, color=INK_2, linewidth=1)
    out = stations[~retained]
    ax.scatter(out.geometry.x, out.geometry.y, s=40, facecolors='none', edgecolors=INK_2, linewidths=1.2,
               label=f'below ratio {ratio} ({len(out)})')
    keep = stations[retained]
    ax.scatter(keep.geometry.x, keep.geometry.y, s=48, color=SERIES_1, edgecolors=SURFACE, linewidths=2,
               label=f'retained ({len(keep)})')
    for _, r in keep.iterrows():
        ax.annotate(r.code, (r.geometry.x, r.geometry.y), xytext=(4, 3), textcoords='offset points', fontsize=7,
                    color=INK_1)
    ax.set_xlabel('longitude', color=INK_2)
    ax.set_ylabel('latitude', color=INK_2)
    ax.set_title(f'INMET stations in the region, control ratio {ratio}', color=INK_1, loc='left')
    ax.tick_params(colors=INK_2)
    for s in ax.spines.values():
        s.set_color('#d8d7d2')
    ax.set_aspect('equal')
    ax.legend(frameon=False, loc='upper left', bbox_to_anchor=(0, -0.1), ncol=2, labelcolor=INK_1)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def availability_heatmap(path, avail, codes, retained):
    """avail: {var: (stations, years) fraction observed}."""
    cmap = LinearSegmentedColormap.from_list('blue', BLUE_RAMP)
    years = avail['years']
    panels = [('u', 'wind (u, v)'), ('temp', 'temperature'), ('dewpoint', 'dewpoint')]
    fig, axes = plt.subplots(1, len(panels), figsize=(9, 0.18 * len(codes) + 1.6), sharey=True, facecolor=SURFACE,
                             layout='constrained')
    labels = [f'{c} ●' if k else c for c, k in zip(codes, retained)]
    for ax, (var, title) in zip(axes, panels):
        im = ax.imshow(avail[var], cmap=cmap, vmin=0, vmax=1, aspect='auto', interpolation='nearest')
        ax.set_title(title, color=INK_1, fontsize=10, loc='left')
        ax.set_xticks(range(len(years)), [str(y)[2:] for y in years], fontsize=8, color=INK_2)
        ax.set_yticks(range(len(codes)), labels, fontsize=7, color=INK_2)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cb.set_label('fraction of hours observed', color=INK_2)
    cb.outline.set_visible(False)
    fig.suptitle('Observed fraction per station and year (● = retained)', color=INK_1, ha='left', x=0.01)
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches='tight')
    plt.close(fig)


def md_table(df, floatfmt='{:.2f}'):
    """Markdown table without the tabulate dependency."""
    fmt = lambda x: floatfmt.format(x) if isinstance(x, float) else str(x)
    rows = [' | '.join(map(str, df.columns)), ' | '.join('---' for _ in df.columns)]
    rows += [' | '.join(fmt(x) for x in r) for r in df.itertuples(index=False)]
    return '\n'.join(f'| {r} |' for r in rows)


def r2(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(np.corrcoef(a[ok], b[ok])[0, 1] ** 2) if ok.sum() > 2 else np.nan


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--years', nargs='+', type=int, default=[2019, 2020, 2021, 2022, 2023])
    parser.add_argument('--control-ratio', type=float, default=0.6)
    parser.add_argument('--shapefile', default='Shapefiles/Regions/ne_brazil_buffered.shp',
                        help='relative to <data-root>/brazil')
    parser.add_argument('--era5-region', default='NEBrazil')
    parser.add_argument('--out', type=Path, help='report folder (default <data-root>/brazil/report)')
    args = parser.parse_args(argv)

    years = sorted(args.years)
    start, end = years[0], years[-1]
    data_path = args.data_root / 'brazil'
    out = args.out or data_path / 'report'
    out.mkdir(parents=True, exist_ok=True)
    check = Checks()

    shapefile = data_path / args.shapefile
    lon_low, lat_low, lon_up, lat_up = gpd.read_file(shapefile).bounds.iloc[0]
    region = gpd.read_file(shapefile).dissolve().geometry.iloc[0]
    file_name = Paths.region_file_name(lon_low, lon_up, lat_low, lat_up, shapefile)
    filtered = Paths.filtered_file_name(file_name, args.control_ratio)
    all_shp = Paths.station_file(data_path, start, end, file_name)
    filt_shp = Paths.station_file(data_path, start, end, filtered)
    for f in (all_shp, filt_shp):
        if not f.exists():
            log.error('%s missing; run inmet_to_localized.py --control-ratio %s first', f, args.control_ratio)
            return 1
    stations = gpd.read_file(all_shp)
    kept = gpd.read_file(filt_shp)
    retained = stations.geometry.isin(list(kept.geometry)).to_numpy()

    # counts
    n_hours = len(pd.date_range(f'{start}-01-01', f'{end}-12-31 23:00', freq='h'))
    counts = pd.DataFrame({'control ratio': RATIOS, 'stations': [int((stations.num >= n_hours * r).sum()) for r in RATIOS]})
    check('station count', len(kept) >= MIN_STATIONS, f'{len(kept)} at ratio {args.control_ratio} (target >= {MIN_STATIONS})')
    stations_map(out / 'stations_map.png', region, stations, retained, args.control_ratio)

    # schema + I1, I2, I4 on the filtered files; availability on the all-station files
    avail = {'years': years, 'u': [], 'temp': [], 'dewpoint': []}
    obs = []
    for y in years:
        f = Paths.madis_processed_file(data_path, start, end, y, filtered)
        if not f.exists():
            check(f'{y} filtered file', False, f'{f.name} missing')
            continue
        with xr.open_dataset(f) as ds:
            ds = ds.load()
        expected_time = pd.date_range(f'{y}-01-01', f'{y}-12-31 23:00', freq='h')
        missing = [v for v in FILLED + [f'{v}_is_real' for v in FILLED] + ['elv', 'lon', 'lat'] if v not in ds]
        check(f'{y} schema', not missing and list(ds.stations.values) == list(range(1, len(kept) + 1)),
              f'missing {missing}' if missing else f'{ds.sizes["stations"]} stations')
        check(f'{y} I1 station order', len(ds.lon) == len(kept)
              and np.array_equal(np.round(ds.lon.values, 4), np.round(kept.geometry.x.values, 4))
              and np.array_equal(np.round(ds.lat.values, 4), np.round(kept.geometry.y.values, 4)))
        check(f'{y} I2 time axis', pd.DatetimeIndex(ds.time.values).equals(expected_time), f'{ds.sizes["time"]} hours')
        nan = [v for v in FILLED if np.isnan(ds[v].values).any()]
        check(f'{y} I4 no NaN', not nan, ', '.join(nan))
        obs.append(ds)
        with xr.open_dataset(Paths.madis_processed_file(data_path, start, end, y, file_name)) as raw:
            for v in ('u', 'temp', 'dewpoint'):
                avail[v].append(raw[f'{v}_is_real'].mean('time').values)
    for v in ('u', 'temp', 'dewpoint'):
        avail[v] = np.stack(avail[v], axis=1) if avail[v] else np.zeros((len(stations), 0))
    if avail['u'].size:
        availability_heatmap(out / 'availability.png', avail, list(stations.code), retained)

    # ERA5 (I3, I5) and station-vs-ERA5 comparison
    diurnal, fig1 = pd.DataFrame(), pd.DataFrame()
    era = era5_at_stations(data_path / 'ERA5' / args.era5_region, years,
                           kept.geometry.x.values, kept.geometry.y.values, check)
    if era is not None and obs:
        st = xr.concat(obs, dim='time')
        era = era.reindex(time=st.time)
        t_obs = st.temp.where(st.temp_is_real)
        diurnal = pd.DataFrame({'code': list(kept.code), 'INMET peak (UTC)': peak_hour(t_obs),
                                'ERA5 peak (UTC)': peak_hour(era.t2m)})
        diurnal['gap (h)'] = hour_gap(diurnal['INMET peak (UTC)'], diurnal['ERA5 peak (UTC)'])
        off = diurnal[diurnal['gap (h)'] > 1]
        check('I3 diurnal peak INMET vs ERA5 within ±1 h', off.empty,
              ', '.join(f'{r.code} ({r["gap (h)"]} h)' for _, r in off.iterrows()) or f'{len(diurnal)} stations')

        rows = []
        for label, day in [(f'{end}-{d}', f'{end}-{d}') for d in SEASON_DAYS] + [(f'all {start}-{end}', None)]:
            sub_st = st.sel(time=day) if day else st
            sub_era = era.sel(time=day) if day else era
            row = {'period': label}
            for v, e in PAIRS.items():
                real = sub_st[f'{v}_is_real'].values
                row[v] = r2(np.where(real, sub_st[v].values, np.nan).ravel(), sub_era[e].values.T.ravel())
            rows.append(row)
        fig1 = pd.DataFrame(rows)

    # report
    lines = [
        '# NE Brazil dataset report', '',
        f'Generated by `scripts/brazil/validate_dataset.py`, years {start}-{end}, control ratio {args.control_ratio}, '
        f'region `{file_name}`.', '',
        f'**Result: {"PASS" if not check.failed else f"FAIL ({len(check.failed)} checks)"}.** '
        f'{len(kept)} of {len(stations)} stations retained at control ratio {args.control_ratio}.', '',
        '## Checks', '', '| check | result | detail |', '|---|---|---|',
        *[f'| {n} | {"PASS" if ok else "**FAIL**"} | {d} |' for n, ok, d in check.rows], '',
        '## Stations per control ratio', '', md_table(counts, '{:.1f}'), '',
        '![stations](stations_map.png)', '',
        '## Availability', '',
        'Fraction of hours with an observation, per station (all stations in the region) and year, before filling. '
        'Wind counts the hours where both u and v exist.', '',
        '![availability](availability.png)', '',
    ]
    if not diurnal.empty:
        lines += ['## Diurnal temperature peak (timezone, I3)', '',
                  'Hour of the mean diurnal temperature maximum, observed values only, vs ERA5 at the nearest grid '
                  'point. Local time is UTC-3.', '',
                  md_table(diurnal['gap (h)'].value_counts().sort_index().rename('stations').rename_axis('gap (h)')
                  .reset_index()), '',
                  '<details><summary>per station</summary>', '', md_table(diurnal), '', '</details>', '']
    if not fig1.empty:
        lines += ['## ERA5-nearest vs INMET, ρ² (paper Fig. 1 analogue)', '',
                  'Squared Pearson correlation over all retained stations and hours of the period, observed values '
                  'only. Low ρ² on single days is the local signal ERA5 misses.', '',
                  md_table(fig1), '']
    (out / 'brazil_data_report.md').write_text('\n'.join(lines), encoding='utf-8')
    log.info('wrote %s', out / 'brazil_data_report.md')

    if check.failed:
        log.error('%d checks failed: %s', len(check.failed), ', '.join(n for n, _, _ in check.failed))
        return 1
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
