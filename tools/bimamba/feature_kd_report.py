"""PURE intra (paper protocol): RhythmMamba-loss teacher / no-KD / output-KD / feature-KD students, plus the\nearlier BiMamba-loss runs for reference.

From the rPPG-Toolbox root:  python tools/bimamba/feature_kd_report.py [--seeds 100 200 300]
Reads each run's saved_test_outputs/paper_protocol_metrics.json; writes runs/exp/feature_kd_report.csv.
Also prints the per-subject MAE (PURE test subjects 7-10) and, for feature-KD runs, the last-epoch
feature-loss parts from training_log if present.
"""
import argparse
import csv
import json
import os

import numpy as np

ROWS = {  # BiMamba-loss rows = earlier reference runs; RM rows = this experiment
    'teacher_bm': ('runs/exp/pure_pinned_dual12_2stage_speedaug', '[BiMamba loss] dual 12+12 teacher (3.16 M)'),
    'nokd_bm': ('runs/exp/pure_pinned_student6_single_nokd_speedaug', '[BiMamba loss] student (0.84 M), no KD'),
    'outkd_bm': ('runs/exp/pure_pinned_student6_single_kd_speedaug', '[BiMamba loss] student, output KD'),
    'teacher': ('runs/exp/pure_pinned_dual12_2stage_rm_speedaug', '[RM loss] dual 12+12 teacher'),
    'nokd': ('runs/exp/pure_pinned_student6_single_nokd_rm_speedaug', '[RM loss] student, no KD'),
    'outkd': ('runs/exp/pure_pinned_student6_single_outkd_rm_speedaug', '[RM loss] student, output KD'),
    'fkd_at': ('runs/exp/pure_pinned_student6_single_fkd_at_rm_speedaug', '[RM loss] student, feature KD: attention transfer'),
    'fkd_hint': ('runs/exp/pure_pinned_student6_single_fkd_hint_rm_speedaug', '[RM loss] student, feature KD: block + final hints'),
    'fkd_athint': ('runs/exp/pure_pinned_student6_single_fkd_athint_rm_speedaug', '[RM loss] student, feature KD: AT + hints'),
}
TEST_EXP = 'PURE_128_c160_std'
METRICS = ('MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR')


def run_dir(base, seed):
    return (base if seed == 100 else base + '_seed%d' % seed) + '/' + TEST_EXP + '/saved_test_outputs'


def load(folder):
    path = os.path.join(folder, 'paper_protocol_metrics.json')
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        m = json.load(handle)
    out = {k: m[k] for k in METRICS}
    per = os.path.join(folder, 'paper_protocol_per_subject.csv')
    if os.path.isfile(per):
        with open(per) as handle:
            out['subjects'] = {r['subject']: abs(float(r['hr_pred']) - float(r['hr_label'])) for r in csv.DictReader(handle)}
    return out


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return '-'
    if len(values) == 1:
        return '%.3f' % values[0]
    return '%.3f±%.3f' % (np.mean(values), np.std(values, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[100, 200, 300])
    ap.add_argument('--out', default='runs/exp/feature_kd_report.csv')
    args = ap.parse_args()
    table = []
    for key, (base, desc) in ROWS.items():
        runs = {s: load(run_dir(base, s)) for s in args.seeds}
        done = {s: r for s, r in runs.items() if r}
        row = dict(run=key, description=desc, seeds_done=' '.join(map(str, done)),
                   seeds_missing=' '.join(str(s) for s, r in runs.items() if not r))
        for k in METRICS:
            row[k] = fmt([r[k] for r in done.values()])
        subjects = sorted({s for r in done.values() for s in r.get('subjects', {})})
        row['subject_mae'] = ' '.join('%s:%s' % (s, fmt([r['subjects'][s] for r in done.values() if s in r.get('subjects', {})]))
                                      for s in subjects) or '-'
        table.append(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print('PURE intra (pinned 1-6 / 7-10, speed aug), paper protocol, mean±sd over seeds done')
    print('DualPhysMamba reference (paper, single run): MAE 0.18  RMSE 0.31\n')
    for row in table:
        print('%-11s %s  [seeds %s%s]' % (row['run'], row['description'], row['seeds_done'] or 'none',
                                         ('; missing ' + row['seeds_missing']) if row['seeds_missing'] else ''))
        print('   MAE %s  RMSE %s  MAPE %s  Pearson %s  SNR %s' % tuple(row[k] for k in METRICS))
        print('   per-video |err|: %s\n' % row['subject_mae'])
    print('Saved', args.out)


if __name__ == '__main__':
    main()
