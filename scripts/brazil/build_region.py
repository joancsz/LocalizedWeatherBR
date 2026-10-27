"""Build the NE Brazil study-region shapefile from IBGE state boundaries.

usage: python scripts/brazil/build_region.py --data-root $DATA_ROOT [--states CE RN PB PE]
writes <data-root>/brazil/Shapefiles/Regions/ne_brazil_buffered.shp (EPSG:4326, one dissolved row)
"""

import argparse
import gzip
import io
import logging
import sys
import urllib.request
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

# IBGE malhas API (the plan's fallback; geobr would add a dependency for the same polygons)
IBGE_URL = ('https://servicodados.ibge.gov.br/api/v3/malhas/estados/{uf}'
            '?formato=application/vnd.geo%2Bjson&periodo=2020&qualidade=maxima')
BUFFER_DEG = 0.1  # mirrors northeastern_buffered

log = logging.getLogger('build_region')


def fetch_state(uf):
    with urllib.request.urlopen(IBGE_URL.format(uf=uf), timeout=60) as r:
        body = r.read()
    # IBGE gzips the body even when not asked to
    return gpd.read_file(io.BytesIO(gzip.decompress(body) if body[:2] == b'\x1f\x8b' else body))


def build(states, bbox):
    parts = [fetch_state(uf) for uf in states]
    gdf = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs='EPSG:4326')
    # bbox clip drops offshore islands (e.g. Fernando de Noronha, PE) that would stretch the ERA5 window
    region = gdf.dissolve().geometry.iloc[0].intersection(box(*bbox))
    region = region.buffer(BUFFER_DEG)
    return gpd.GeoDataFrame({'name': ['_'.join(states)]}, geometry=[region], crs='EPSG:4326')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--years', nargs='*', type=int, help='unused; accepted for a uniform CLI')
    parser.add_argument('--states', nargs='+', default=['CE', 'RN', 'PB', 'PE'])
    parser.add_argument('--bbox', nargs=4, type=float, default=[-41.5, -9.5, -34.5, -2.5],
                        metavar=('LON_MIN', 'LAT_MIN', 'LON_MAX', 'LAT_MAX'))
    parser.add_argument('--name', default='ne_brazil_buffered')
    args = parser.parse_args(argv)

    out = args.data_root / 'brazil' / 'Shapefiles' / 'Regions' / f'{args.name}.shp'
    if out.exists():
        region = gpd.read_file(out)
        log.info('%s exists, skipping', out)
    else:
        try:
            region = build(args.states, args.bbox)
        except Exception as e:
            log.error('could not build region: %s', e)
            return 1
        if region.geometry.iloc[0].is_empty or not region.geometry.iloc[0].is_valid:
            log.error('region polygon is empty or invalid')
            return 1
        out.parent.mkdir(parents=True, exist_ok=True)
        region.to_file(out)
        log.info('wrote %s', out)

    lon_low, lat_low, lon_up, lat_up = region.total_bounds
    log.info('bounds lon [%.4f, %.4f] lat [%.4f, %.4f] -> file_name stem %.2f_%.2f_%.2f_%.2f%s',
             lon_low, lon_up, lat_low, lat_up, lon_low, lon_up, lat_low, lat_up, out.stem)
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
