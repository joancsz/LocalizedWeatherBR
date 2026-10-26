"""Download and extract the NE-US Zenodo data (record 15346612) into <data-root>/neus/."""

import argparse
import hashlib
import logging
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

URL = 'https://zenodo.org/records/15346612/files/WindDataNE-US.zip?download=1'
MD5 = '6d95a6e785d7249d697cec29ed6d7c67'

log = logging.getLogger('download_zenodo')


def md5sum(path):
    h = hashlib.md5()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, retries=3):
    part = dest.with_suffix('.part')
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(part, 'wb') as f:
                size = int(r.headers.get('Content-Length', 0))
                for chunk in iter(lambda: r.read(1 << 20), b''):
                    f.write(chunk)
            if size and part.stat().st_size != size:
                raise IOError(f'size mismatch: got {part.stat().st_size}, expected {size}')
            part.replace(dest)
            return
        except Exception as e:
            log.warning('attempt %d/%d failed: %s', attempt, retries, e)
            if attempt == retries:
                raise
            time.sleep(10 * attempt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--out', type=Path, help='extraction dir (default: <data-root>/neus)')
    args = parser.parse_args(argv)
    out = args.out or args.data_root / 'neus'

    existing = sorted(out.glob('**/madis/processed/Meta-2019-2023'))
    if existing:
        log.info('data already present at %s, skipping', existing[0].parent.parent.parent)
        return 0

    out.mkdir(parents=True, exist_ok=True)
    zip_path = out / 'WindDataNE-US.zip'
    if not (zip_path.exists() and md5sum(zip_path) == MD5):
        log.info('downloading %s', URL)
        download(URL, zip_path)
    if md5sum(zip_path) != MD5:
        log.error('md5 mismatch for %s', zip_path)
        return 1

    log.info('extracting to %s', out)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)

    roots = sorted(out.glob('**/madis/processed/Meta-2019-2023'))
    if not roots:
        log.error('no madis/processed/Meta-2019-2023 found after extraction')
        return 1
    log.info('set data_path to %s', roots[0].parent.parent.parent)
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
