import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts' / 'eval'))
import make_tables


def row(entries, frac, seed, n, wind):
    return dict(entries=entries, keep_fraction=frac, seed=seed, n_stations=n, n_eval=6, wind=wind, t_rmse=1.0,
                td_rmse=2.0)


def test_density_mean_sd_over_seeds_and_references():
    evals = pd.DataFrame([row('model', 1.0, 0, 26, 0.5), row('persistence', 1.0, 0, 26, 0.4),
                          row('ERA5', 1.0, 0, 26, 2.0), row('model', 0.5, 0, 13, 0.6), row('model', 0.5, 1, 13, 0.8)])
    tex = make_tables.e4_density_table(evals)
    assert r'\toprule' in tex and r'\bottomrule' in tex
    assert '1.00 & 26 & 1 & 0.500' in tex
    assert '0.50 & 13 & 2 & 0.700 (0.141)' in tex          # mean and sample sd of 0.6, 0.8
    assert 'persistence &  &  & 0.400' in tex
    assert tex.index('1.00 &') < tex.index('0.50 &') < tex.index('persistence')


def test_ci_cell_and_escape():
    r = pd.Series(dict(wind=0.5, wind_lo=0.45, wind_hi=0.55, t_rmse=1.0))
    assert make_tables.ci(r, 'wind') == r'0.500 {\scriptsize[0.450, 0.550]}'
    assert make_tables.ci(r, 't_rmse') == '1.000'
    assert r'sparse\_gaps' in make_tables.degraded_table({'sparse_gaps': r.to_dict() | {'td_rmse': 1.0}})


def test_lead_table_rows_by_lead():
    df = lambda w: pd.DataFrame([dict(entries='model', wind=w, t_rmse=1.0, td_rmse=1.0),
                                 dict(entries='persistence', wind=2 * w, t_rmse=1.0, td_rmse=1.0)])
    tex = make_tables.lead_table({6: df(0.6), 1: df(0.4)}, 'NE-US')
    assert tex.index('1 & model & 0.400') < tex.index('1 & persistence & 0.800') < tex.index('6 & model & 0.600')
    assert 'tab:leads_neus' in tex
    assert 'lead 6 h' in make_tables.degraded_table({'base': df(0.6).iloc[0].to_dict()}, 6, 26)
