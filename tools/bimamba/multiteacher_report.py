"""UBFC-rPPG -> PURE: multi-teacher KD students vs the same student without KD (ablation B), single-teacher
HINT KD, and the teachers themselves. From the rPPG-Toolbox root:
    python tools/bimamba/multiteacher_report.py [--seeds 100 200 300]
    python tools/bimamba/multiteacher_report.py --check     # teachers present? which checkpoint is used?

Stop rule (README): keep multi-teacher KD only if it lowers MAE AND the doubled-video count below
ablation B across the seeds.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import yaml

TEST = 'PURE_128_c160_std'
METRICS = ('MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR')
ROWS = [
    ('student, no KD (ablation B)', 'runs/exp/ablationB_cross_ubfc2pure_student6_single_nokd'),
    ('student, HINT KD (our BiMamba teacher)', 'runs/exp/cross_ubfc2pure_student6_single_fkd_hint_rm_speedaug'),
    ('student, multi-teacher KD: 3 DPM', 'runs/exp/cross_ubfc2pure_student6_single_mtkd3dpm_rm_speedaug'),
    ('student, multi-teacher KD: 3 DPM + 3 RM', 'runs/exp/cross_ubfc2pure_student6_single_mtkd6_rm_speedaug'),
    ('student, multi-teacher KD: 3 DPM + 3 RM + 3 RF', 'runs/exp/cross_ubfc2pure_student6_single_mtkd9_rm_speedaug'),
    ('teacher: DualPhysMamba repro', 'runs/exp/cross_ubfc2pure_dualphysmamba_rm_speedaug'),
    ('teacher: RhythmMamba', 'runs/exp/cross_ubfc2pure_rhythmmamba_rm_speedaug'),
    ('teacher: RhythmFormer', 'runs/exp/cross_ubfc2pure_rhythmformer_rm_speedaug'),
]


def load(base, seed):
    folder = (base if seed == 100 else base + '_seed%d' % seed) + '/' + TEST + '/saved_test_outputs'
    path = os.path.join(folder, 'paper_protocol_metrics.json')
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        m = json.load(handle)
    with open(os.path.join(folder, 'paper_protocol_per_subject.csv')) as handle:
        rows = [(r['subject'], float(r['hr_label']), float(r['hr_pred'])) for r in csv.DictReader(handle)]
    bad = [(v, g, p) for v, g, p in rows if abs(p - g) > 3]
    return dict(m={k: m[k] for k in METRICS}, bad=bad, doubled=sum(1 for _, g, p in bad if 1.8 < p / g < 2.2))


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return '-'
    return '%.3f' % values[0] if len(values) == 1 else '%.3f±%.3f' % (np.mean(values), np.std(values, ddof=1))


def check():
    sys.path.insert(0, os.getcwd())
    from neural_methods.trainer.BiMambaPhysKDTrainer import resolve_teacher_run
    ok = True
    for v in ('MT6', 'MT3DPM', 'MT9'):
        with open('configs/train_configs/UBFC_PURE_BIMAMBA_STUDENT_SINGLE_%s_RM_SPEEDAUG_CROSS.yaml' % v) as handle:
            runs = yaml.safe_load(handle)['DISTILL']['TEACHER_RUNS']
        print(v)
        for run in runs:
            try:
                ckpt, _ = resolve_teacher_run(run)
                print('   ok  ', run, '->', os.path.basename(str(ckpt)))
            except Exception as error:  # report every missing teacher, not just the first
                ok = False
                print('   MISSING', run, ':', error)
    sys.exit(0 if ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[100, 200, 300])
    ap.add_argument('--check', action='store_true')
    ap.add_argument('--out', default='runs/exp/multiteacher_report.csv')
    args = ap.parse_args()
    if args.check:
        return check()
    print('UBFC-rPPG -> PURE, paper protocol (mean±sd over seeds done); DualPhysMamba paper: MAE 0.63 RMSE 1.13')
    print('Prediction-averaging reference (not a single model): 3 DPM + 3 RM runs averaged = MAE 0.40, 0 doubled\n')
    table = []
    for name, base in ROWS:
        done = {s: r for s in args.seeds for r in [load(base, s)] if r}
        row = dict(model=name, folder=base, seeds_done=' '.join(map(str, done)))
        for k in METRICS:
            row[k] = fmt([r['m'][k] for r in done.values()])
        row['per_seed_MAE'] = ' '.join('%.2f' % r['m']['MAE'] for r in done.values()) or '-'
        row['bad_videos_per_seed'] = ' '.join(str(len(r['bad'])) for r in done.values()) or '-'
        row['doubled_per_seed'] = ' '.join(str(r['doubled']) for r in done.values()) or '-'
        row['bad_videos'] = ' | '.join(' '.join('%s(%d->%d)' % (v, g, p) for v, g, p in r['bad']) or '-' for r in done.values())
        table.append(row)
        print('%-42s [%s]' % (name, row['seeds_done'] or 'none'))
        print('   MAE %s  RMSE %s  MAPE %s  Pearson %s  SNR %s' % tuple(row[k] for k in METRICS))
        print('   per-seed MAE %s | >3 bpm %s | doubled %s' % (row['per_seed_MAE'], row['bad_videos_per_seed'], row['doubled_per_seed']))
        if done:
            print('   %s' % row['bad_videos'])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print('\nSaved', args.out)


if __name__ == '__main__':
    main()
