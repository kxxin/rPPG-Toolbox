"""RhythmMamba-loss suite: teacher / no-KD / HINT feature-KD students in every setting, with the earlier
BiMamba-loss runs and DualPhysMamba for reference.

From the rPPG-Toolbox root:  python tools/bimamba/rm_suite_report.py [--seeds 100 200 300]
Reads saved_test_outputs/paper_protocol_metrics.json (+ per-subject CSV) of each run; writes
runs/exp/rm_suite_report.csv. Also reports the number of videos with |error| > 3 bpm and, of those,
'doubled' videos (pred/GT 1.8-2.2) per seed.
"""
import argparse
import csv
import json
import os

import numpy as np

METRICS = ('MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR')
SETTINGS = [  # name, test EXP_DATA_NAME, DualPhysMamba (paper) or None, rows
    ('UBFC intra', 'experiment00', None, [
        ('teacher RM', 'runs/exp/pinned_dual12_2stage_rm_speedaug'),
        ('student no-KD RM', 'runs/exp/pinned_student6_single_nokd_rm_speedaug'),
        ('student HINT RM', 'runs/exp/pinned_student6_single_fkd_hint_rm_speedaug'),
        ('(BiMamba) teacher', 'runs/exp/pinned_dual12_2stage_speedaug'),
        ('(BiMamba) student no-KD', 'runs/exp/pinned_student6_single_nokd_speedaug'),
        ('(BiMamba) student out-KD', 'runs/exp/pinned_student6_single_kd_speedaug')]),
    ('PURE intra', 'PURE_128_c160_std', (0.18, 0.31, None, None, None), [
        ('teacher RM', 'runs/exp/pure_pinned_dual12_2stage_rm_speedaug'),
        ('student no-KD RM', 'runs/exp/pure_pinned_student6_single_nokd_rm_speedaug'),
        ('student HINT RM', 'runs/exp/pure_pinned_student6_single_fkd_hint_rm_speedaug'),
        ('(BiMamba) teacher', 'runs/exp/pure_pinned_dual12_2stage_speedaug'),
        ('(BiMamba) student no-KD', 'runs/exp/pure_pinned_student6_single_nokd_speedaug')]),
    ('PURE -> UBFC', 'experiment00', (0.86, 1.60, 0.89, 0.99, 6.86), [
        ('teacher RM', 'runs/exp/cross_pure2ubfc_dual12_2stage_rm_speedaug'),
        ('student no-KD RM', 'runs/exp/cross_pure2ubfc_student6_single_nokd_rm_speedaug'),
        ('student HINT RM', 'runs/exp/cross_pure2ubfc_student6_single_fkd_hint_rm_speedaug'),
        ('(BiMamba) teacher', 'runs/exp/cross_pure2ubfc_dual12_2stage_speedaug'),
        ('(BiMamba) student no-KD', 'runs/exp/cross_pure2ubfc_student6_single_nokd_speedaug')]),
    ('UBFC -> PURE', 'PURE_128_c160_std', (0.63, 1.13, 0.93, 0.99, 10.84), [
        ('teacher RM', 'runs/exp/cross_ubfc2pure_dual12_2stage_rm_speedaug'),
        ('student no-KD RM (abl. B)', 'runs/exp/ablationB_cross_ubfc2pure_student6_single_nokd'),
        ('student HINT RM', 'runs/exp/cross_ubfc2pure_student6_single_fkd_hint_rm_speedaug'),
        ('(BiMamba) teacher', 'runs/exp/cross_ubfc2pure_dual12_2stage_speedaug'),
        ('(BiMamba) student no-KD', 'runs/exp/cross_ubfc2pure_student6_single_nokd_speedaug')]),
]


def load(base, seed, test_exp):
    folder = (base if seed == 100 else base + '_seed%d' % seed) + '/' + test_exp + '/saved_test_outputs'
    path = os.path.join(folder, 'paper_protocol_metrics.json')
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        m = json.load(handle)
    out = {k: m[k] for k in METRICS}
    out['big'] = out['doubled'] = None
    per = os.path.join(folder, 'paper_protocol_per_subject.csv')
    if os.path.isfile(per):
        with open(per) as handle:
            rows = [(float(r['hr_label']), float(r['hr_pred'])) for r in csv.DictReader(handle)]
        err = np.array([p - g for g, p in rows]); ratio = np.array([p / g for g, p in rows])
        out['big'] = int(np.sum(np.abs(err) > 3))
        out['doubled'] = int(np.sum((np.abs(err) > 3) & (ratio > 1.8) & (ratio < 2.2)))
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
    ap.add_argument('--out', default='runs/exp/rm_suite_report.csv')
    args = ap.parse_args()
    table = []
    for setting, test_exp, paper, rows in SETTINGS:
        print('\n== %s (paper protocol, mean±sd over seeds done)' % setting)
        if paper:
            print('   DualPhysMamba (paper): ' + '  '.join('%s %s' % (k, v) for k, v in zip(METRICS, paper) if v is not None))
        for name, base in rows:
            done = {s: r for s in args.seeds for r in [load(base, s, test_exp)] if r}
            row = dict(setting=setting, run=name, folder=base, seeds_done=' '.join(map(str, done)))
            for k in METRICS:
                row[k] = fmt([r[k] for r in done.values()])
            row['big_err_videos_per_seed'] = ' '.join(str(r['big']) for r in done.values() if r['big'] is not None) or '-'
            row['doubled_per_seed'] = ' '.join(str(r['doubled']) for r in done.values() if r['doubled'] is not None) or '-'
            table.append(row)
            print('   %-26s [%s]  MAE %s  RMSE %s  MAPE %s  Pearson %s  SNR %s  | >3bpm %s  doubled %s'
                  % (name, row['seeds_done'] or 'none', *(row[k] for k in METRICS),
                     row['big_err_videos_per_seed'], row['doubled_per_seed']))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print('\nSaved', args.out)


if __name__ == '__main__':
    main()
