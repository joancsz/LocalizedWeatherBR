import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'brazil'))
import inmet_parser as P

META = ('REGI?O:;NE\nUF:;CE\nESTAC?O:;FORTALEZA\nCODIGO (WMO):;A305\nLATITUDE:;-3,815701\n'
        'LONGITUDE:;-38,537792\nALTITUDE:;29,55\nDATA DE FUNDAC?O:;18/02/03\n')
COLS = ('RADIACAO GLOBAL (KJ/m²);TEMPERATURA DO AR - BULBO SECO, HORARIA (°C);'
        'TEMPERATURA DO PONTO DE ORVALHO (°C);VENTO, DIREÇÃO HORARIA (gr) (° (gr));VENTO, VELOCIDADE HORARIA (m/s);')

NEW = META + 'Data;Hora UTC;' + COLS + '\n' + (
    '2019/01/01;0000 UTC;;27,1;23;90;2;\n'
    '2019/01/01;0100 UTC;3600;27,2;22,9;180;0;\n'
    '2019/01/01;0200 UTC;;;;;;\n')
OLD = META + 'DATA (YYYY-MM-DD);HORA (UTC);' + COLS + '\n' + (
    '2018-01-01;00:00;-9999;27,1;23;90;2;\n'
    '2018-01-01;01:00;3600;27,2;22,9;180;0;\n'
    '2018-01-01;02:00;-9999;-9999;-9999;-9999;-9999;\n')


@pytest.mark.parametrize('text,year', [(NEW, 2019), (OLD, 2018)])
def test_parse_both_header_variants(tmp_path, text, year):
    f = tmp_path / 'x.CSV'
    f.write_bytes(text.encode('latin-1'))
    meta, header, df = P.parse_file(f)
    assert meta['code'] == 'A305' and meta['uf'] == 'CE'
    assert meta['lat'] == pytest.approx(-3.815701) and meta['elv'] == pytest.approx(29.55)
    assert list(df.index) == list(pd.date_range(f'{year}-01-01', periods=3, freq='h'))
    assert df.temp.iloc[0] == pytest.approx(27.1) and np.isnan(df.solar.iloc[0])
    assert df.iloc[2].isna().all()


def test_qc_ranges_and_flat_line():
    t = pd.date_range('2019-01-01', periods=10, freq='h')
    df = pd.DataFrame({'ws': [60, 1, 2, 0, 0, 0, 0, 0, 0, 3.0], 'wd': [10.0] * 10,
                       'temp': [25, 26, 27, 28, 29, 30, 31, 32, 33, 99.0],
                       'dewpoint': [30, 20, 20, 20, 20, 20, 20, 21, 22, 23.0],
                       'solar': [-1, 0, 0, 0, 0, 0, 0, 5, 6, 7.0]}, index=t)
    out, frac = P.qc(df)
    assert np.isnan(out.ws.iloc[0]) and np.isnan(out.temp.iloc[9])
    assert np.isnan(out.dewpoint.iloc[0])            # dewpoint > temp + 0.5
    assert out.ws.iloc[3:9].eq(0).all()              # calm run kept
    assert out.solar.iloc[1:7].eq(0).all()           # night zero radiation kept
    assert out.dewpoint.iloc[1:7].isna().all()       # 6 identical hours rejected
    assert out.wd.iloc[[0, 1, 2, 9]].isna().all() and out.wd.iloc[3:9].eq(10).all()  # flat wd excused only when calm
    assert frac['temp'] == pytest.approx(0.1)


def test_to_localized_units_and_calm():
    df = pd.DataFrame({'ws': [2.0, 0.0], 'wd': [270.0, np.nan], 'temp': [0.0, 10.0],
                       'dewpoint': [-1.0, 5.0], 'solar': [3600.0, 0.0]})
    out = P.to_localized(df)
    assert out.u.iloc[0] == pytest.approx(2.0) and out.v.iloc[0] == pytest.approx(0.0, abs=1e-12)  # wind from W
    assert out.u.iloc[1] == 0 and out.v.iloc[1] == 0  # calm with missing direction
    assert out.temp.iloc[0] == pytest.approx(273.15) and out.solar_radiation.iloc[0] == pytest.approx(1000.0)


def test_check_timezone():
    t = pd.date_range('2019-01-01', periods=48, freq='h')
    utc_peak = pd.Series(np.cos(2 * np.pi * (t.hour - 17) / 24), index=t)
    peaks = P.diurnal_peak_hours({'A': utc_peak})
    assert P.check_timezone(peaks) == 17
    with pytest.raises(ValueError):
        P.check_timezone(P.diurnal_peak_hours({'A': utc_peak.shift(-6)}))
