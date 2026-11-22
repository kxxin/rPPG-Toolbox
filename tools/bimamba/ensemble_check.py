"""Prediction-averaging check: would these teachers' errors cancel? (no training; reads saved test outputs)

From the rPPG-Toolbox root:
    python tools/bimamba/ensemble_check.py                      # UBFC->PURE, default teacher sets
    python tools/bimamba/ensemble_check.py --setting pure2ubfc
    python tools/bimamba/ensemble_check.py --sets DPM,RM DPM,RM,RF RF

For each model: every seed alone and the average of its seeds. For each set: the average of ALL seeds of
all models in the set (what multi-teacher KD distils) and the seed-matched averages. Averaging = per-video
mean of z-scored full-video predictions, then the paper protocol. Same procedure as the analysis that gave
MAE 0.40 for DPM+RM on UBFC->PURE.

Use it to decide the teacher set BEFORE training students: add a model only if the set average gets
better (fewer large-error videos / lower MAE) with it.
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

sys.path.insert(0, os.getcwd())
import evaluation.paper_protocol as pp  # noqa: E402


def _fast_detrend(signal, lambda_value=100):  # identical result to the dense inverse, much faster
    n = signal.shape[0]
    d = sparse.diags([1, -2, 1], [0, 1, 2], shape=(n - 2, n))
    return signal - spsolve((sparse.identity(n) + lambda_value ** 2 * (d.T @ d)).tocsc(), signal)


_x = np.random.RandomState(0).randn(300)
assert np.allclose(pp._detrend(_x), _fast_detrend(_x), atol=1e-8)
pp._detrend = _fast_detrend

SETTINGS = {'ubfc2pure': ('cross_ubfc2pure', 'PURE_128_c160_std'), 'pure2ubfc': ('cross_pure2ubfc', 'experiment00'),
            'ubfc': ('pinned', 'experiment00'), 'pure': ('pure_pinned', 'PURE_128_c160_std')}
MODELS = {'DPM': 'dualphysmamba', 'RM': 'rhythmmamba', 'RF': 'rhythmformer', 'RFLR': 'rhythmformer_lr9e3',
          'BIDPM': 'bidualphysmamba_v2', 'OURS': 'dual12_2stage'}
SEEDS = (100, 200, 300)
z = lambda x: (x - x.mean()) / (x.std() + 1e-8)


def load(prefix, model, test):
    out = {}
    for seed in SEEDS:
        run = 'runs/exp/%s_%s_rm_speedaug%s' % (prefix, MODELS[model], '' if seed == 100 else '_seed%d' % seed)
        files = glob.glob(run + '/' + test + '/saved_test_outputs/*.pickle')
        if files:
            with open(files[0], 'rb') as handle:
                d = pickle.load(handle)
            out[seed] = ({v: pp.reform(c) for v, c in d['predictions'].items()},
                         {v: pp.reform(c) for v, c in d['labels'].items()})
    return out


def score(preds, labels):
    vids = sorted(preds, key=pp.subject_sort_key)
    rows = [pp.metric_per_video(preds[v], labels[v]) for v in vids]
    err = np.array([r[1] - r[0] for r in rows]); gt = np.array([r[0] for r in rows]); pr = np.array([r[1] for r in rows])
    bad = [v for v, e in zip(vids, err) if abs(e) > 3]
    return 'MAE %6.3f  RMSE %6.3f  SNR %6.2f | >3bpm %2d (doubled %d) %s' % (
        np.abs(err).mean(), np.sqrt((err ** 2).mean()), np.mean([r[2] for r in rows]), len(bad),
        int(((abs(err) > 3) & (pr / gt > 1.8)).sum()), ' '.join(bad))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--setting', default='ubfc2pure', choices=sorted(SETTINGS))
    ap.add_argument('--sets', nargs='+', default=['DPM,RM', 'DPM,RM,RF', 'DPM,RF', 'RM,RF'])
    args = ap.parse_args()
    prefix, test = SETTINGS[args.setting]
    wanted = sorted({m for s in args.sets for m in s.split(',')})
    data = {m: load(prefix, m, test) for m in wanted}
    print('== %s: prediction averages (paper protocol)' % args.setting)
    for m, runs in data.items():
        if not runs:
            print('  %-6s no runs found (runs/exp/%s_%s_rm_speedaug*)' % (m, prefix, MODELS[m]))
            continue
        labels = next(iter(runs.values()))[1]
        for seed, (pred, _) in runs.items():
            print('  %-6s seed %d      %s' % (m, seed, score(pred, labels)), flush=True)
        print('  %-6s avg %d seeds  %s' % (m, len(runs), score({v: np.mean([z(r[0][v]) for r in runs.values()], 0) for v in labels}, labels)), flush=True)
    for s in args.sets:
        members = s.split(',')
        if any(not data[m] for m in members):
            print('  %-12s skipped (missing runs)' % s)
            continue
        labels = next(iter(data[members[0]].values()))[1]
        allruns = [r for m in members for r in data[m].values()]
        print('  %-12s ALL %d runs   %s' % (s, len(allruns), score({v: np.mean([z(r[0][v]) for r in allruns], 0) for v in labels}, labels)), flush=True)
        for seed in SEEDS:
            if all(seed in data[m] for m in members):
                print('  %-12s seed %d      %s' % (s, seed, score({v: np.mean([z(data[m][seed][0][v]) for m in members], 0) for v in labels}, labels)), flush=True)


if __name__ == '__main__':
    main()
