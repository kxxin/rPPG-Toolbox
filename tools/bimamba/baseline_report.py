"""Baselines (reference DualPhysMamba reproduction, RhythmMamba) vs our RhythmMamba-loss models, all under the
same protocol. From the rPPG-Toolbox root:  python tools/bimamba/baseline_report.py [--seeds 100 200 300]

Per setting:
  1. mean±sd of the 5 paper-protocol metrics, and videos with |error| > 3 bpm per seed;
  2. hard videos: |error| per model (mean over seeds) on the videos our models get wrong
     (PURE->UBFC subjects 11/23/37, UBFC->PURE PURE subject 5 + 904), and any other video that is
     wrong in some run;
  3. complementarity with our teacher, seed by seed: videos the baseline gets right (<= 3 bpm) that our
     teacher gets wrong, and vice versa. A multi-teacher setup can only help if this is non-zero.
Writes runs/exp/baseline_report.csv (metrics) and runs/exp/baseline_hard_videos.csv.
"""
import argparse
import csv
import json
import os

import numpy as np

METRICS = ('MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR')
SETTINGS = [  # name, test EXP_DATA_NAME, folder prefix, DualPhysMamba paper numbers or None
    ('UBFC intra', 'experiment00', 'pinned', None),
    ('PURE intra', 'PURE_128_c160_std', 'pure_pinned', (0.18, 0.31, None, None, None)),
    ('PURE -> UBFC', 'experiment00', 'cross_pure2ubfc', (0.86, 1.60, 0.89, 0.99, 6.86)),
    ('UBFC -> PURE', 'PURE_128_c160_std', 'cross_ubfc2pure', (0.63, 1.13, 0.93, 0.99, 10.84)),
]
HARD = {'PURE -> UBFC': ['subject11', 'subject23', 'subject37'],
        'UBFC -> PURE': ['501', '502', '503', '504', '505', '506', '904']}


def models(prefix):
    our_student = ('runs/exp/ablationB_cross_ubfc2pure_student6_single_nokd' if prefix == 'cross_ubfc2pure'
                   else 'runs/exp/%s_student6_single_nokd_rm_speedaug' % prefix)
    return [('ours: teacher (BiMamba dual)', 'runs/exp/%s_dual12_2stage_rm_speedaug' % prefix),
            ('ours: student HINT', 'runs/exp/%s_student6_single_fkd_hint_rm_speedaug' % prefix),
            ('ours: student no-KD', our_student),
            ('DualPhysMamba repro (uni)', 'runs/exp/%s_dualphysmamba_rm_speedaug' % prefix),
            ('RhythmMamba', 'runs/exp/%s_rhythmmamba_rm_speedaug' % prefix),
            ('RhythmFormer', 'runs/exp/%s_rhythmformer_rm_speedaug' % prefix),
            ('RhythmFormer (own LR 9e-3)', 'runs/exp/%s_rhythmformer_lr9e3_rm_speedaug' % prefix),
            ('DualPhysMamba repro + Bi (v1)', 'runs/exp/%s_bidualphysmamba_rm_speedaug' % prefix),
            ('DualPhysMamba repro + Bi', 'runs/exp/%s_bidualphysmamba_v2_rm_speedaug' % prefix)]


def load(base, seed, test_exp):
    folder = (base if seed == 100 else base + '_seed%d' % seed) + '/' + test_exp + '/saved_test_outputs'
    path = os.path.join(folder, 'paper_protocol_metrics.json')
    if not os.path.isfile(path):
        return None
    with open(path) as handle:
        m = json.load(handle)
    with open(os.path.join(folder, 'paper_protocol_per_subject.csv')) as handle:
        err = {r['subject']: float(r['hr_pred']) - float(r['hr_label']) for r in csv.DictReader(handle)}
    return dict(metrics={k: m[k] for k in METRICS}, err=err)


def fmt(values):
    values = [v for v in values if v is not None]
    if not values:
        return '-'
    return '%.3f' % values[0] if len(values) == 1 else '%.3f±%.3f' % (np.mean(values), np.std(values, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='+', type=int, default=[100, 200, 300])
    ap.add_argument('--out', default='runs/exp/baseline_report.csv')
    ap.add_argument('--hard-out', default='runs/exp/baseline_hard_videos.csv')
    args = ap.parse_args()
    table, hard_rows = [], []
    for setting, test_exp, prefix, paper in SETTINGS:
        print('\n' + '=' * 100 + '\n%s (paper protocol, mean±sd over seeds done)' % setting)
        if paper:
            print('  DualPhysMamba paper (single run): ' + '  '.join('%s %s' % (k, v) for k, v in zip(METRICS, paper) if v is not None))
        runs = {}
        for name, base in models(prefix):
            done = {s: r for s in args.seeds for r in [load(base, s, test_exp)] if r}
            runs[name] = done
            row = dict(setting=setting, model=name, folder=base, seeds_done=' '.join(map(str, done)))
            for k in METRICS:
                row[k] = fmt([r['metrics'][k] for r in done.values()])
            row['big_err_videos_per_seed'] = ' '.join(str(sum(abs(e) > 3 for e in r['err'].values())) for r in done.values()) or '-'
            table.append(row)
            print('  %-30s [%-11s] MAE %-13s RMSE %-13s MAPE %-13s Pearson %-13s SNR %-13s | >3bpm %s'
                  % (name, row['seeds_done'] or 'none', *(row[k] for k in METRICS), row['big_err_videos_per_seed']))
        # hard videos
        wrong_anywhere = {v for done in runs.values() for r in done.values() for v, e in r['err'].items() if abs(e) > 3}
        videos = sorted(set(HARD.get(setting, [])) | wrong_anywhere)
        if videos:
            print('  hard videos, mean |error| over seeds (bpm):')
            print('  %-30s ' % '' + ' '.join('%9s' % v[-9:] for v in videos))
            for name, done in runs.items():
                if not done:
                    continue
                vals = [np.mean([abs(r['err'][v]) for r in done.values() if v in r['err']]) if any(v in r['err'] for r in done.values()) else np.nan
                        for v in videos]
                print('  %-30s ' % name + ' '.join('%9.1f' % x for x in vals))
                hard_rows.append(dict(setting=setting, model=name, **{v: '%.2f' % x for v, x in zip(videos, vals)}))
        # complementarity vs our teacher (same seed)
        ours = runs['ours: teacher (BiMamba dual)']
        for name in ('DualPhysMamba repro (uni)', 'RhythmMamba', 'RhythmFormer', 'RhythmFormer (own LR 9e-3)', 'DualPhysMamba repro + Bi'):
            pairs = [(s, ours[s], runs[name][s]) for s in args.seeds if s in ours and s in runs[name]]
            if not pairs:
                continue
            fixes = [sorted(v for v, e in o['err'].items() if abs(e) > 3 and abs(b['err'].get(v, 99)) <= 3) for _, o, b in pairs]
            breaks = [sorted(v for v, e in o['err'].items() if abs(e) <= 3 and abs(b['err'].get(v, 0)) > 3) for _, o, b in pairs]
            print('  %-30s vs our teacher, per seed: right where ours is wrong %s | wrong where ours is right %s'
                  % (name, [len(f) for f in fixes], [len(b) for b in breaks]))
            if any(fixes):
                print('  %-30s   videos it fixes: %s' % ('', ' | '.join(' '.join(f) or '-' for f in fixes)))
        uni, bi = runs['DualPhysMamba repro (uni)'], runs['DualPhysMamba repro + Bi']
        pairs = [s for s in args.seeds if s in uni and s in bi]
        if pairs:
            d = {k: [bi[s]['metrics'][k] - uni[s]['metrics'][k] for s in pairs] for k in METRICS}
            print('  Bi - Uni (same seed; direction is the only change): ' + '  '.join(
                '%s %+.3f' % (k, np.mean(v)) for k, v in d.items()) + '   per-seed MAE %s' % ['%+.3f' % x for x in d['MAE']])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    if hard_rows:
        fields = ['setting', 'model'] + sorted({k for r in hard_rows for k in r} - {'setting', 'model'})
        with open(args.hard_out, 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(hard_rows)
    print('\nSaved', args.out, 'and', args.hard_out)


if __name__ == '__main__':
    main()
