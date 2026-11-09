"""LaTeX (booktabs) tables of E2, E3 (summary), E4 and the degraded NE-US experiment, from the evaluation CSVs.

usage: python scripts/eval/make_tables.py --data-root $DATA_ROOT [--out $DATA_ROOT/tables]

Reads (run names as in notebooks 03 and 04; a table whose inputs are missing is skipped with a warning):
  E2     brazil/ModelOutputs/ne_brazil_{vit,gnn}/baselines.csv
  E3     brazil/ModelOutputs/ne_brazil_vit/diurnal/wind_error.csv
  E4     brazil/ModelOutputs/{ne_brazil_vit,ne_brazil_e4_frac*}/eval_fixed.csv
         brazil/ModelOutputs/{ne_brazil_vit,ne_brazil_e4_ratio*}/eval_ratio0.8.csv
  leads  brazil/ModelOutputs/ne_brazil_vit*/baselines.csv, neus/ModelOutputs/neus_deg_base*/baselines.csv
  extra  neus/ModelOutputs/neus_deg_*/ test pickles (model only), one table per lead, scored with common_set.py
Leads come from each run's params.json.
Every number is on real observations only; brackets are 95% day-bootstrap intervals.
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common_set

METRICS = [('wind', 'wind (m/s)'), ('t_rmse', 'T RMSE (K)'), ('td_rmse', 'Td RMSE (K)')]
DEGRADED = ['base', 'sparse', 'gaps', 'coarse_050', 'coarse_100', 'sparse_gaps', 'all']  # notebook 03 runs; '_l<h>' = lead h
CAVEAT = 'With the grid at the target hour, these are reanalysis downscaling results.'

log = logging.getLogger('make_tables')


def ci(row, m):
    if f'{m}_lo' not in row or pd.isna(row.get(f'{m}_lo')):
        return f'{row[m]:.3f}'
    return f'{row[m]:.3f} {{\\scriptsize[{row[m + "_lo"]:.3f}, {row[m + "_hi"]:.3f}]}}'


def booktabs(header, rows, caption, label, align=None):
    align = align or 'l' + 'r' * (len(header) - 1)
    lines = ['\\begin{table}[ht]', '\\centering', f'\\caption{{{caption}}}', f'\\label{{{label}}}',
             f'\\begin{{tabular}}{{{align}}}', '\\toprule', ' & '.join(header) + ' \\\\', '\\midrule']
    lines += [' & '.join(str(c) for c in r) + ' \\\\' for r in rows]
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(lines) + '\n'


def lead_of(run_dir):
    return int(json.load(open(run_dir / 'params.json'))['lead_hrs'])


def e2_table(vit, gnn=None, lead=1):
    """vit / gnn: baselines.csv frames (rows model, ERA5, persistence)."""
    by = lambda df, e: df.set_index('entries').loc[e]
    rows = [('Transformer + ERA5', by(vit, 'model'))]
    if gnn is not None:
        rows.append(('GNN + ERA5', by(gnn, 'model')))
    rows += [('ERA5 nearest', by(vit, 'ERA5')), ('Persistence', by(vit, 'persistence'))]
    return booktabs(['Forecast'] + [h for _, h in METRICS], [[n] + [ci(r, m) for m, _ in METRICS] for n, r in rows],
                    f'E2: NE Brazil test year, lead {lead} h. {CAVEAT}', 'tab:e2')


def e3_table(err, counts):
    """err: diurnal wind_error.csv; counts: {group: n stations}."""
    rows = []
    for (g, f), d in err.groupby(['group', 'forecast'], sort=False):
        peak = d.loc[d.wind.idxmax()]
        rows.append([f'{g} ({counts.get(g, "?")})', f, f'{d.wind.mean():.3f}', f'{peak.wind:.3f}', int(peak.hour)])
    return booktabs(['Stations', 'Forecast', 'Mean (m/s)', 'Max (m/s)', 'Peak hour'], rows,
                    'E3: wind-vector error over the local day (UTC$-$3), coastal ($<$20 km) vs inland.', 'tab:e3',
                    'llrrr')


def e4_density_table(evals):
    """evals: concatenated eval_fixed.csv frames. Model mean (sd) over seeds per fraction, plus the persistence and
    ERA5 references, which do not depend on the fraction."""
    model = evals[evals.entries == 'model']
    rows = []
    for frac, d in model.groupby('keep_fraction', sort=False):
        cells = [f'{d[m].mean():.3f}' + (f' ({d[m].std():.3f})' if len(d) > 1 else '') for m, _ in METRICS]
        rows.append([f'{frac:.2f}', int(d.n_stations.iloc[0]), len(d)] + cells)
    rows.sort(key=lambda r: -float(r[0]))
    for ref in ('persistence', 'ERA5'):
        r = evals[evals.entries == ref].iloc[0]
        rows.append([ref, '', ''] + [f'{r[m]:.3f}' for m, _ in METRICS])
    n_eval = int(evals.n_eval.iloc[0])
    return booktabs(['Fraction', 'Stations', 'Seeds'] + [h for _, h in METRICS], rows,
                    f'E4: station density, scored on the same {n_eval} fixed stations; mean (sd) over seeds.',
                    'tab:e4density')


def e4_filter_table(evals):
    model = evals[evals.entries == 'model'].sort_values('control_ratio')
    rows = [[f'{r.control_ratio:.1f}', int(r.n_stations)] + [ci(r, m) for m, _ in METRICS] for _, r in model.iterrows()]
    n_eval = int(evals.n_eval.iloc[0])
    return booktabs(['Ratio', 'Stations'] + [h for _, h in METRICS], rows,
                    f'E4: availability filter, scored on the {n_eval} stations kept at ratio 0.8.', 'tab:e4filter')


def degraded_table(results, lead=1, n_stations=None):
    """results: {run: summarize row} for the notebook 03 runs of one lead, scored on common stations and hours."""
    rows = [[n.replace('_', '\\_')] + [ci(r, m) for m, _ in METRICS] for n, r in results.items()]
    on = f', every run scored on the same {n_stations} stations and real hours' if n_stations else ''
    return booktabs(['Run'] + [h for _, h in METRICS], rows,
                    f'Degraded NE-US (Transformer + ERA5, lead {lead} h{on}). {CAVEAT}', f'tab:degraded_l{lead}')


def lead_table(frames, region):
    """frames: {lead: baselines.csv frame}. Model, ERA5 and persistence per lead."""
    rows = [[lead, r.entries] + [ci(r, m) for m, _ in METRICS]
            for lead in sorted(frames) for _, r in frames[lead].iterrows()]
    return booktabs(['Lead (h)', 'Forecast'] + [h for _, h in METRICS], rows,
                    f'{region}: Transformer + ERA5 and references by lead. {CAVEAT}',
                    f'tab:leads_{region.lower().replace("-", "")}', 'rl' + 'r' * len(METRICS))


def degraded_order(run_dir):
    name = re.sub(r'_l\d+$', '', run_dir.name.removeprefix('neus_deg_'))
    return DEGRADED.index(name) if name in DEGRADED else len(DEGRADED)


def flat_ci(row):
    return {k2: v2 for k, v in row.items() for k2, v2 in
            (zip((k[:-3] + '_lo', k[:-3] + '_hi'), v) if k.endswith('_ci') else [(k, v)])}


def build(root):
    br, us = root / 'brazil' / 'ModelOutputs', root / 'neus' / 'ModelOutputs'
    read = lambda p: pd.read_csv(p) if p.exists() else None
    tables = {}

    vit, gnn = read(br / 'ne_brazil_vit' / 'baselines.csv'), read(br / 'ne_brazil_gnn' / 'baselines.csv')
    if vit is not None:
        tables['e2'] = e2_table(vit, gnn, lead_of(br / 'ne_brazil_vit'))

    for key, region, dirs in (('leads_brazil', 'Brazil', br.glob('ne_brazil_vit*')),
                              ('leads_neus', 'NE-US', us.glob('neus_deg_base*'))):
        frames = {lead_of(d): pd.read_csv(d / 'baselines.csv') for d in dirs if (d / 'baselines.csv').exists()}
        if frames:
            tables[key] = lead_table(frames, region)

    err = read(br / 'ne_brazil_vit' / 'diurnal' / 'wind_error.csv')
    st = read(br / 'ne_brazil_vit' / 'diurnal' / 'stations.csv')
    if err is not None and st is not None:
        tables['e3'] = e3_table(err, st.group.value_counts().to_dict())

    for key, fname, glob, fn in (('e4_density', 'eval_fixed.csv', 'ne_brazil_e4_frac*', e4_density_table),
                                 ('e4_filter', 'eval_ratio0.8.csv', 'ne_brazil_e4_ratio*', e4_filter_table)):
        frames = [read(d / fname) for d in [br / 'ne_brazil_vit', *sorted(br.glob(glob))]]
        frames = [f for f in frames if f is not None]
        if frames:
            tables[key] = fn(pd.concat(frames, ignore_index=True))

    by_lead = {}
    for d in sorted(us.glob('neus_deg_*'), key=degraded_order):
        if (d / 'Preds_CUSTOM_min.pkl').exists():
            by_lead.setdefault(lead_of(d), []).append(d)
    for lead, dirs in sorted(by_lead.items()):
        ref = min(dirs, key=lambda d: len(common_set.station_xy(d)))  # the sparsest network's stations
        rows = common_set.common_scores(dirs, ref)
        tables[f'degraded_l{lead}'] = degraded_table(
            {n.removeprefix('neus_deg_'): flat_ci(r) for n, r in rows.items()}, lead, len(common_set.station_xy(ref)))
    return tables


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    out = args.out or args.data_root / 'tables'
    out.mkdir(parents=True, exist_ok=True)
    tables = build(args.data_root)
    for name in ('e2', 'e3', 'e4_density', 'e4_filter', 'leads_brazil', 'leads_neus', 'degraded_l1'):
        if name not in tables:
            log.warning('%s: inputs missing, skipped', name)
    for name, tex in tables.items():
        (out / f'{name}.tex').write_text(tex, encoding='utf-8')
        log.info('wrote %s', out / f'{name}.tex')
    return 0 if tables else 1


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    sys.exit(main())
