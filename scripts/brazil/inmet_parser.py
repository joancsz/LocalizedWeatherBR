"""Parse, QC and convert INMET automatic-station CSVs (annual archive format).

Format seen in the 2019 archive: latin-1, ';' separated, ',' decimals, 8 metadata lines (accented letters in the
keys are literal '?'), then a header row with 'Data' (YYYY/MM/DD) and 'Hora UTC' (HHMM UTC). Missing values are
empty strings; older files use -9999 and 'DATA (YYYY-MM-DD)' / 'HORA (UTC)' (HH:MM). Both are handled.
"""

import io
import re
import unicodedata

import numpy as np
import pandas as pd

COLUMNS = {
    'ws': 'VENTO, VELOCIDADE HORARIA',
    'wd': 'VENTO, DIRECAO HORARIA',
    'temp': 'TEMPERATURA DO AR - BULBO SECO',
    'dewpoint': 'TEMPERATURA DO PONTO DE ORVALHO',
    'solar': 'RADIACAO GLOBAL',
}
META_KEYS = {'UF': 'uf', 'ESTAC': 'name', 'CODIGO': 'code', 'LATITUDE': 'lat', 'LONGITUDE': 'lon', 'ALTITUDE': 'elv'}
SENTINEL = -9999
FLAT_HOURS = 6


def normalise(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()
    return re.sub(r'\s+', ' ', s).strip().upper()


def _number(s):
    # ',916877' (A302) and '-3,815701' both parse
    try:
        return float(s.strip().replace(',', '.'))
    except ValueError:
        return np.nan


def _is_header(line):
    fields = [normalise(f) for f in line.split(';')[:2]]
    return len(fields) == 2 and fields[0].startswith('DATA') and fields[1].startswith('HORA')


def read_meta(path, max_lines=20):
    """Station metadata from the lines above the header; cheap, for region pre-filtering."""
    meta = {}
    with open(path, encoding='latin-1') as f:
        for _, line in zip(range(max_lines), f):
            if _is_header(line):
                break
            key, _, value = line.partition(';')
            key = normalise(key)
            for prefix, name in META_KEYS.items():
                if key.startswith(prefix) and name not in meta:
                    meta[name] = value.strip()
    for name in ('lat', 'lon', 'elv'):
        meta[name] = _number(meta.get(name, ''))
    if 'code' not in meta:
        raise ValueError(f'{path}: no CODIGO in metadata')
    return meta


def parse_file(path):
    """Return (meta, header_line, df) with df indexed by naive-UTC hourly time and columns ws, wd, temp, dewpoint,
    solar in file units (m/s, deg, degC, degC, kJ/m2 per hour), NaN where missing."""
    with open(path, encoding='latin-1') as f:
        lines = f.read().splitlines()
    i = next((k for k, line in enumerate(lines) if _is_header(line)), None)
    if i is None:
        raise ValueError(f'{path}: no Data/Hora header row')
    meta = read_meta(path)

    raw = pd.read_csv(io.StringIO('\n'.join(lines[i:])), sep=';', dtype=str, keep_default_na=False)
    names = [normalise(c) for c in raw.columns]

    date = pd.to_datetime(raw.iloc[:, 0].str.strip().str.replace('/', '-'), format='%Y-%m-%d', errors='coerce')
    hour = pd.to_numeric(raw.iloc[:, 1].str.extract(r'^\s*(\d{1,2}):?\d{2}')[0], errors='coerce')
    time = date + pd.to_timedelta(hour, unit='h')

    data = {}
    for target, needle in COLUMNS.items():
        j = next((k for k, n in enumerate(names) if needle in n), None)
        if j is None:
            raise ValueError(f'{path}: no column matching {needle!r}')
        values = pd.to_numeric(raw.iloc[:, j].str.strip().str.replace(',', '.'), errors='coerce')
        data[target] = values.mask(values == SENTINEL).to_numpy()

    df = pd.DataFrame(data, index=pd.DatetimeIndex(time, name='time'))
    df = df[df.index.notna()]
    return meta, lines[i], df[~df.index.duplicated()].sort_index()


def _flat_runs(s, exclude):
    """True where s holds the same non-NaN value for >= FLAT_HOURS consecutive hours, outside `exclude`."""
    run_id = (s != s.shift()).cumsum()
    run_len = s.groupby(run_id).transform('size')
    return (run_len >= FLAT_HOURS) & s.notna() & ~exclude


def qc(df):
    """Set implausible observations to NaN. Returns (clean df, fraction of observed values rejected per column)."""
    out = df.copy()
    out['ws'] = out.ws.mask((out.ws < 0) | (out.ws > 50))
    out['wd'] = out.wd.mask((out.wd < 0) | (out.wd > 360))
    out['temp'] = out.temp.mask((out.temp < -5) | (out.temp > 50))
    out['dewpoint'] = out.dewpoint.mask((out.dewpoint < -20) | (out.dewpoint > 35) | (out.dewpoint > out.temp + 0.5))
    out['solar'] = out.solar.mask(out.solar < 0)

    # calm hours and night-time zero radiation legitimately repeat
    exclude = {'ws': out.ws == 0, 'wd': out.ws == 0, 'solar': out.solar == 0}
    for c in out.columns:
        out[c] = out[c].mask(_flat_runs(out[c], exclude.get(c, pd.Series(False, index=out.index))))

    observed = df.notna().sum()
    rejected = (df.notna() & out.isna()).sum()
    return out, (rejected / observed.where(observed > 0)).fillna(0.0)


def to_localized(df):
    """MADIS units and variables: u, v (m/s), temp, dewpoint (K), solar_radiation (W/m2)."""
    ws, wd = df.ws, df.wd
    # same expression as Madis.load_madis_hourly
    u = np.cos(np.deg2rad(270 - wd)) * ws
    v = np.sin(np.deg2rad(270 - wd)) * ws
    calm = ws == 0  # direction is meaningless when calm
    return pd.DataFrame({
        'u': u.mask(calm, 0.0),
        'v': v.mask(calm, 0.0),
        'temp': df.temp + 273.15,
        'dewpoint': df.dewpoint + 273.15,
        'solar_radiation': df.solar / 3.6,
    }, index=df.index)


def diurnal_peak_hours(temps):
    """UTC hour of the mean diurnal temperature maximum per station. temps: {code: Series indexed by time}."""
    return pd.Series({code: s.groupby(s.index.hour).mean().idxmax() for code, s in temps.items() if s.notna().any()})


def check_timezone(peaks, lo=14, hi=20):
    """Fail loudly unless the median peak hour is in [lo, hi] UTC (12-16 local, UTC-3)."""
    median = float(np.median(peaks))
    if not lo <= median <= hi:
        raise ValueError(f'median diurnal temperature peak at {median} UTC, outside [{lo}, {hi}]: '
                         f'timestamps are probably not UTC')
    return median
