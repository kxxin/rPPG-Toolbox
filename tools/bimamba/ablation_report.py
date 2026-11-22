"""Compare ablation variants A-E on UBFC-rPPG -> PURE (single-path no-KD student), seeds 100/200/300.

From the rPPG-Toolbox root:  python tools/bimamba/ablation_report.py [--seeds 100 200 300]
Reads each run's saved_test_outputs (paper protocol) directly; writes runs/exp/ablation_report.csv.

Per variant: mean +/- sd of MAE, RMSE, MAPE, Pearson, SNR; MAE on low-HR videos (GT < 55 bpm) and
on the rest; number of 'doubled' videos per seed (|error| > 3 bpm and pred/GT in 1.8-2.2);
waveform Pearson (macro subject mean, waveform_summary.json).
"""
import argparse
import csv
import json
import os

import numpy as np

VARIANTS = {
    'A': ('runs/exp/cross_ubfc2pure_student6_single_nokd_speedaug', 'BiMamba loss, clip-target aug p0.5'),
    'B': ('runs/exp/ablationB_cross_ubfc2pure_student6_single_nokd', 'RhythmMamba loss, clip-target aug p0.5'),
    'C': ('runs/exp/ablationC_cross_ubfc2pure_student6_single_nokd', 'BiMamba loss, HR-balanced aug p1.0'),
    'D': ('runs/exp/ablationD_cross_ubfc2pure_student6_single_nokd', 'RhythmMamba loss, HR-balanced aug p1.0'),
    'E': ('runs/exp/ablationE_cross_ubfc2pure_student6_single_nokd', 'RhythmMamba loss + RhythmMamba aug (2x slow/fast, flip)'),
}
TEST_EXP = 'PURE_128_c160_std'
METRICS = ('MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR')


def run_dir(base, seed):
    return (base if seed == 100 else base + '_seed%d' % seed) + '/' + TEST_EXP + '/saved_test_outputs'


def load(folder):
    metrics_path = os.path.join(folder, 'paper_protocol_metrics.json')
    if not os.path.isfile(metrics_path):
        return None
    with open(metrics_path) as handle:
        m = json.load(handle)
    with open(os.path.join(folder, 'paper_protocol_per_subject.csv')) as handle:
        rows = [dict(subject=r['subject'], gt=float(r['hr_label']), pred=float(r['hr_pred'])) for r in csv.DictReader(handle)]
    wave = None
    if os.path.isfile(os.path.join(folder, 'waveform_summary.json')):
        with open(os.path.join(folder, 'waveform_summary.json')) as handle:
            wave = json.load(handle)['macro_subject_mean']['pearson']
    err = np.array([r['pred'] - r['gt'] for r in rows]); gt = np.array([r['gt'] for r in rows])
    ratio = np.array([r['pred'] / r['gt'] for r in rows])
    low = gt < 55
    return dict(**{k: m[k] for k in METRICS}, waveform_r=wave,
                doubled=int(np.sum((np.abs(err) > 3) & (ratio > 1.8) & (ratio < 2.2))),
                big_errors=int(np.sum(np.abs(err) > 3)),
                mae_low=float(np.mean(np.abs(err[low]))) if low.any() else float('nan'),
                mae_rest=float(np.mean(np.abs(err[~low]))), n_low=int(low.sum()), n=len(rows))


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
    ap.add_argument('--out', default='runs/exp/ablation_report.csv')
    args = ap.parse_args()
    table = []
    for variant, (base, desc) in VARIANTS.items():
        runs = {s: load(run_dir(base, s)) for s in args.seeds}
        done = {s: r for s, r in runs.items() if r}
        missing = [s for s, r in runs.items() if not r]
        row = dict(variant=variant, description=desc, seeds_done=' '.join(map(str, done)),
                   seeds_missing=' '.join(map(str, missing)))
        for k in METRICS + ('mae_low', 'mae_rest', 'waveform_r'):
            row[k] = fmt([r[k] for r in done.values()])
        row['doubled_per_seed'] = ' '.join(str(r['doubled']) for r in done.values()) or '-'
        row['big_errors_per_seed'] = ' '.join(str(r['big_errors']) for r in done.values()) or '-'
        table.append(row)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print('UBFC-rPPG -> PURE, single-path no-KD student, paper protocol (mean±sd over seeds done)')
    print('DualPhysMamba reference: MAE 0.63  RMSE 1.13  MAPE 0.93  Pearson 0.99  SNR 10.84\n')
    for row in table:
        print('%s  %s  [seeds %s%s]' % (row['variant'], row['description'], row['seeds_done'] or 'none',
                                       ('; missing ' + row['seeds_missing']) if row['seeds_missing'] else ''))
        print('   MAE %s  RMSE %s  MAPE %s  Pearson %s  SNR %s' % tuple(row[k] for k in METRICS))
        print('   MAE GT<55 bpm %s | GT>=55 %s | doubled videos per seed: %s (all |err|>3: %s) | waveform r %s\n'
              % (row['mae_low'], row['mae_rest'], row['doubled_per_seed'], row['big_errors_per_seed'], row['waveform_r']))
    print('Saved', args.out)


if __name__ == '__main__':
    main()
