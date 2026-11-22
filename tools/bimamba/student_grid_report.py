"""Student grid: {our single-path BiMamba student, compact multi-temporal student} x {BiMamba, RhythmMamba,
combined loss}, no KD, all four settings; plus the 5 M reproductions for reference.
From the rPPG-Toolbox root:  python tools/bimamba/student_grid_report.py [--seeds 100 200 300]
Writes runs/exp/student_grid_report.csv.
"""
import argparse
import csv
import json
import os

import numpy as np

METRICS = ('MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR')
SETTINGS = [('UBFC intra', 'pinned', 'experiment00'), ('PURE intra', 'pure_pinned', 'PURE_128_c160_std'),
            ('PURE -> UBFC', 'cross_pure2ubfc', 'experiment00'), ('UBFC -> PURE', 'cross_ubfc2pure', 'PURE_128_c160_std')]


def rows(prefix):
    ours_rm = ('runs/exp/ablationB_cross_ubfc2pure_student6_single_nokd' if prefix == 'cross_ubfc2pure'
               else 'runs/exp/%s_student6_single_nokd_rm_speedaug' % prefix)
    return [('ours      x BiMamba loss', 'runs/exp/%s_student6_single_nokd_speedaug' % prefix),
            ('ours      x RhythmMamba loss', ours_rm),
            ('ours      x combined loss', 'runs/exp/%s_student6_single_nokd_comb_speedaug' % prefix),
            ('compact   x BiMamba loss', 'runs/exp/%s_compact6_bm_speedaug' % prefix),
            ('compact   x RhythmMamba loss', 'runs/exp/%s_compact6_rm_speedaug' % prefix),
            ('compact   x combined loss', 'runs/exp/%s_compact6_comb_speedaug' % prefix),
            ('[ref] DualPhysMamba repro (RM loss)', 'runs/exp/%s_dualphysmamba_rm_speedaug' % prefix),
            ('[ref] RhythmMamba repro (RM loss)', 'runs/exp/%s_rhythmmamba_rm_speedaug' % prefix)]


def load(base, seed, test):
    folder = (base if seed == 100 else base + '_seed%d' % seed) + '/' + test + '/saved_test_outputs'
    path = os.path.join(folder, 'paper_protocol_metrics.json')
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        m = json.load(handle)
    out = {k: m[k] for k in METRICS}
    per = os.path.join(folder, 'paper_protocol_per_subject.csv')
    out['bad'] = out['doubled'] = None
    if os.path.isfile(per):
        with open(per) as handle:
            r = [(float(x['hr_label']), float(x['hr_pred'])) for x in csv.DictReader(handle)]
        out['bad'] = sum(abs(p - g) > 3 for g, p in r)
        out['doubled'] = sum(abs(p - g) > 3 and 1.8 < p / g < 2.2 for g, p in r)
    return out


def params(base):
    import glob
    for log in sorted(glob.glob(base + '/*/main_seed*.log')):
        with open(log, errors='ignore') as handle:
            for line in handle:
                if 'BiMambaPhys parameters' in line:
                    return line.strip().split()[-1]
    return '-'


def fmt(v):
    v = [x for x in v if x is not None]
    if not v:
        return '-'
    return '%.3f' % v[0] if len(v) == 1 else '%.3f±%.3f' % (np.mean(v), np.std(v, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[100, 200, 300])
    ap.add_argument('--out', default='runs/exp/student_grid_report.csv')
    args = ap.parse_args()
    table = []
    for setting, prefix, test in SETTINGS:
        print('\n== %s (paper protocol, mean±sd over seeds done; no KD)' % setting)
        for name, base in rows(prefix):
            done = {s: r for s in args.seeds for r in [load(base, s, test)] if r}
            row = dict(setting=setting, model=name, folder=base, params=params(base), seeds_done=' '.join(map(str, done)))
            for k in METRICS:
                row[k] = fmt([r[k] for r in done.values()])
            row['bad_per_seed'] = ' '.join(str(r['bad']) for r in done.values() if r['bad'] is not None) or '-'
            row['doubled_per_seed'] = ' '.join(str(r['doubled']) for r in done.values() if r['doubled'] is not None) or '-'
            table.append(row)
            print('  %-38s %9s [%-15s] MAE %-13s RMSE %-13s MAPE %-13s r %-13s SNR %-13s | >3bpm %s | doubled %s'
                  % (name, row['params'], row['seeds_done'] or 'none', *(row[k] for k in METRICS),
                     row['bad_per_seed'], row['doubled_per_seed']))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print('\nSaved', args.out)


if __name__ == '__main__':
    main()
