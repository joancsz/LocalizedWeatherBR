"""Download INMET annual station archives and extract them to <data-root>/brazil/inmet/raw/{year}/.

usage: python scripts/brazil/download_inmet.py --data-root $DATA_ROOT [--years 2019 2020 ...]
The first year doubles as the URL check: if it fails, nothing else is tried.
"""

import argparse
import logging
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'neus'))
from download_zenodo import download  # streamed, retries with backoff, Content-Length size check

URL = 'https://portal.inmet.gov.br/uploads/dadoshistoricos/{year}.zip'

log = logging.getLogger('download_inmet')


def fetch_year(year, root):
    raw = root / 'raw' / str(year)
    if any(raw.rglob('*.[Cc][Ss][Vv]')):
        log.info('%d: %s already extracted, skipping', year, raw)
        return
    zip_path = root / 'zips' / f'{year}.zip'
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if not (zip_path.exists() and zipfile.is_zipfile(zip_path)):
        log.info('%d: downloading %s', year, URL.format(year=year))
        download(URL.format(year=year), zip_path)
    with zipfile.ZipFile(zip_path) as z:
        bad = z.testzip()
        if bad is not None:
            raise IOError(f'{zip_path}: corrupt member {bad}')
        names = [n for n in z.namelist() if n.lower().endswith('.csv')]
        if not names:
            raise IOError(f'{zip_path}: no CSV files inside')
        tmp = raw.with_name(f'{year}.tmp')  # rename after extraction so a crash never leaves a half year that looks done
        shutil.rmtree(tmp, ignore_errors=True)
        z.extractall(tmp)
        shutil.rmtree(raw, ignore_errors=True)
        tmp.rename(raw)
    log.info('%d: extracted %d CSV files to %s', year, len(names), raw)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--years', nargs='+', type=int, default=[2019, 2020, 2021, 2022, 2023])
    args = parser.parse_args(argv)

    root = args.data_root / 'brazil' / 'inmet'
    for year in args.years:
        try:
            fetch_year(year, root)
        except Exception as e:
            log.error('%d: %s', year, e)
            return 1
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
