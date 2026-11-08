"""E3: does the model correct ERA5's diurnal (sea-breeze) wind error at coastal stations?

usage: python scripts/eval/diurnal_analysis.py --run-dir <run> [--coast-km 20] [--utc-offset -3]

Writes to <run>/diurnal/:
  stations.csv       distance to the coastline and group (coastal / inland) of every station
  wind_error.csv     wind-vector error by local hour, group and forecast (model, ERA5, persistence), day-bootstrap CI
  uv_cycle.csv       mean u and v by local hour and group: observed, model, ERA5 (same real entries)
  wind_error.png     the main figure: error by local hour, one panel per group
  uv_cycle.png       mean u and v cycle at the coastal stations
Coastline: Natural Earth 10 m (downloaded once to <data_path>/Shapefiles/Coastline/). Real observations only.
"""

import argparse
import json
import logging
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import baselines
import metrics

COASTLINE_URL = 'https://naciscdn.org/naturalearth/10m/physical/ne_10m_coastline.zip'
FORECASTS = ['model', 'ERA5', 'persistence']
# dataviz reference palette: categorical slots 1-3 (validated all-pairs), primary/secondary ink, gridline, surface
COLORS = {'model': '#2a78d6', 'ERA5': '#eb6834', 'persistence': '#1baf7a', 'observed': '#0b0b0b'}
INK_2, MUTED, GRID, SURFACE = '#52514e', '#898781', '#e1e0d9', '#fcfcfb'

log = logging.getLogger('diurnal_analysis')


def local_hours(times, utc_offset):
    """Local hour of day (0-23) of each sample's target hour."""
    return ((metrics.to_datetime(times[:, -1]).hour.values + utc_offset) % 24).astype(int)


def coastline(data_path):
    path = Path(data_path) / 'Shapefiles' / 'Coastline' / 'ne_10m_coastline.zip'
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        log.info('downloading %s', COASTLINE_URL)
        urllib.request.urlretrieve(COASTLINE_URL, path)
    return gpd.read_file(path)


def coast_distance_km(lons, lats, coast):
    """Distance (km) from each station to the nearest coastline segment, in an equidistant projection around them."""
    pts = gpd.GeoSeries(gpd.points_from_xy(lons, lats), crs='EPSG:4326')
    near = coast.cx[lons.min() - 3:lons.max() + 3, lats.min() - 3:lats.max() + 3]
    crs = f'+proj=aeqd +lat_0={np.mean(lats)} +lon_0={np.mean(lons)} +units=m'
    lines = near.to_crs(crs).union_all()
    return pts.to_crs(crs).distance(lines).values / 1000


def wind_error_by_hour(forecasts, targets, is_real, madis_vars_o, hours, days, groups):
    """Rows (group, forecast, hour, wind, wind_lo, wind_hi, n) of the real-only wind-vector error."""
    iu, iv = madis_vars_o.index(metrics.U), madis_vars_o.index(metrics.V)
    mask = is_real[..., iu] & is_real[..., iv]
    rows = []
    for group, st in groups.items():
        for name in FORECASTS:
            err = np.hypot(forecasts[name][..., iu] - targets[..., iu], forecasts[name][..., iv] - targets[..., iv])
            for h in range(24):
                e, m = err[hours == h][:, st], mask[hours == h][:, st]
                if not m.any():
                    continue
                lo, hi = metrics.day_bootstrap_ci(e, m, days[hours == h])
                rows.append(dict(group=group, forecast=name, hour=h, wind=float(e[m].mean()), wind_lo=lo, wind_hi=hi,
                                 n=int(m.sum())))
    return pd.DataFrame(rows)


def uv_cycle(forecasts, targets, is_real, madis_vars_o, hours, groups):
    """Mean u and v by local hour and group for observed, model and ERA5 over the same real entries."""
    iu, iv = madis_vars_o.index(metrics.U), madis_vars_o.index(metrics.V)
    mask = is_real[..., iu] & is_real[..., iv]
    series = {'observed': targets, 'model': forecasts['model'], 'ERA5': forecasts['ERA5']}
    rows = []
    for group, st in groups.items():
        for h in range(24):
            m = mask[hours == h][:, st]
            if not m.any():
                continue
            for name, a in series.items():
                a = a[hours == h][:, st]
                rows.append(dict(group=group, source=name, hour=h, u=float(a[..., iu][m].mean()),
                                 v=float(a[..., iv][m].mean())))
    return pd.DataFrame(rows)


def style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc='left', fontsize=11, color='#0b0b0b')
    ax.set_xlabel('local hour', color=INK_2)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.set_xticks(range(0, 24, 3))
    ax.set_xlim(0, 23)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.tick_params(colors=MUTED)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color('#c3c2b7')


def plot_error(df, counts, path):
    groups = [g for g in ('coastal', 'inland') if g in set(df.group)]
    fig, axes = plt.subplots(1, len(groups), figsize=(5.5 * len(groups), 4), sharey=True, squeeze=False,
                             facecolor=SURFACE)
    for ax, g in zip(axes[0], groups):
        for name in FORECASTS:
            d = df[(df.group == g) & (df.forecast == name)]
            ax.fill_between(d.hour, d.wind_lo, d.wind_hi, color=COLORS[name], alpha=0.15, linewidth=0)
            ax.plot(d.hour, d.wind, color=COLORS[name], linewidth=2, label=name)
            ax.annotate(name, (d.hour.iloc[-1], d.wind.iloc[-1]), xytext=(4, 0), textcoords='offset points',
                        va='center', fontsize=8, color=INK_2)
        style(ax, f'{g} ({counts[g]} stations)', 'wind-vector error (m/s)')
    axes[0][0].set_ylim(bottom=0)
    axes[0][-1].legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_uv(df, n_coastal, path):
    d = df[df.group == 'coastal']
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), facecolor=SURFACE)
    for ax, comp in zip(axes, ('u', 'v')):
        for name in ('observed', 'model', 'ERA5'):
            s = d[d.source == name]
            ax.plot(s.hour, s[comp], color=COLORS[name], linewidth=2, label=name,
                    linestyle='--' if name == 'observed' else '-')
        ax.axhline(0, color='#c3c2b7', linewidth=0.8)
        style(ax, f'mean {comp}, coastal stations ({n_coastal})', f'{comp} (m/s)')
    axes[1].legend(frameon=False, fontsize=9, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--coast-km', type=float, default=20.0, help='coastal if within this distance of the coastline')
    parser.add_argument('--utc-offset', type=int, default=-3, help='local time = UTC + offset (Brazil NE: -3)')
    args = parser.parse_args(argv)
    try:
        forecasts, targets, is_real, madis_vars_o, times, madis = baselines.load_forecasts(args.run_dir)
    except FileNotFoundError as e:
        log.error('missing input: %s', e)
        return 1
    data_path = json.load(open(args.run_dir / 'params.json'))['data_path']
    lons, lats = madis.lon.values, madis.lat.values
    dist = coast_distance_km(lons, lats, coastline(data_path))
    coastal = dist < args.coast_km
    for km in (10, 20, 30, 50):
        log.info('stations within %d km of the coast: %d of %d', km, int((dist < km).sum()), len(dist))
    groups = {g: st for g, st in (('coastal', coastal), ('inland', ~coastal)) if st.any()}
    if len(groups) < 2:
        log.warning('only the %s group has stations at %.0f km', list(groups), args.coast_km)

    out = args.run_dir / 'diurnal'
    out.mkdir(exist_ok=True)
    pd.DataFrame(dict(station=np.arange(1, len(dist) + 1), lon=lons, lat=lats, coast_km=dist,
                      group=np.where(coastal, 'coastal', 'inland'))).to_csv(out / 'stations.csv', index=False)
    hours = local_hours(times, args.utc_offset)
    err = wind_error_by_hour(forecasts, targets, is_real, madis_vars_o, hours, metrics.target_days(times), groups)
    uv = uv_cycle(forecasts, targets, is_real, madis_vars_o, hours, groups)
    err.to_csv(out / 'wind_error.csv', index=False)
    uv.to_csv(out / 'uv_cycle.csv', index=False)
    counts = {g: int(st.sum()) for g, st in groups.items()}
    plot_error(err, counts, out / 'wind_error.png')
    plot_uv(uv, counts.get('coastal', 0), out / 'uv_cycle.png')

    summary = err.groupby(['group', 'forecast']).wind.agg(['mean', 'max', 'idxmax'])
    summary['peak hour'] = err.loc[summary.pop('idxmax'), 'hour'].values
    sys.stdout.write(summary.round(3).to_string() + '\n')
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
