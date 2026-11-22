"""Re-score every saved test output under runs/exp with BOTH protocols (no retraining).

Run from the rPPG-Toolbox root:
    python tools/bimamba/rescore_runs.py                       # all runs under runs/exp
    python tools/bimamba/rescore_runs.py --runs-root runs/exp --out runs/exp/rescore

Writes <out>/rescore_summary.csv and <out>/rescore_per_subject.csv and prints a table.
  toolbox : rPPG-Toolbox post_process (whole-video periodogram, 0.6-3.3 Hz, SNR 10log10)
  paper   : evaluation/paper_protocol.py (RhythmMamba/DualPhysMamba: Welch, 0.75-2.5 Hz,
            SNR 20log10; SNR_10log10 column gives the same ratio in toolbox convention)
The test-subject list is written per run, so runs on different splits are never compared
silently.
"""
import argparse
import csv
import glob
import os
import pickle
import sys

import numpy as np
import scipy.signal
from scipy.signal import butter

sys.path.insert(0, os.getcwd())
from evaluation import post_process as toolbox  # noqa: E402
from evaluation import paper_protocol as paper  # noqa: E402


def toolbox_per_video(pred, label, fs, diff_flag):
    """Same steps as evaluation.post_process.calculate_metric_per_video (FFT), minus MACC."""
    if diff_flag:
        pred, label = toolbox._detrend(np.cumsum(pred), 100), toolbox._detrend(np.cumsum(label), 100)
    else:
        pred, label = toolbox._detrend(pred, 100), toolbox._detrend(label, 100)
    b, a = butter(1, [0.6 / fs * 2, 3.3 / fs * 2], btype='bandpass')
    pred = scipy.signal.filtfilt(b, a, np.double(pred))
    label = scipy.signal.filtfilt(b, a, np.double(label))
    hr_pred, hr_label = toolbox._calculate_fft_hr(pred, fs=fs), toolbox._calculate_fft_hr(label, fs=fs)
    return hr_label, hr_pred, toolbox._calculate_SNR(pred, hr_label, fs=fs)


def score(path):
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
    fs, label_type = data.get('fs', 30), data.get('label_type', 'Standardized')
    diff_flag = label_type == 'DiffNormalized'
    paper_summary, paper_rows = paper.evaluate(data['predictions'], data['labels'], fs, label_type)
    tb_rows = []
    for subject in sorted(data['predictions'], key=paper.subject_sort_key):
        hr_label, hr_pred, snr = toolbox_per_video(paper.reform(data['predictions'][subject]),
                                                   paper.reform(data['labels'][subject]), fs, diff_flag)
        tb_rows.append(dict(subject=str(subject), hr_label=float(hr_label), hr_pred=float(hr_pred),
                            error=float(hr_pred - hr_label), snr_db20=float(2 * snr), snr_db10=float(snr)))
    tb_summary = paper.summarize(tb_rows)
    # Toolbox reports its native 10log10 SNR as "SNR".
    tb_summary['SNR'], tb_summary['SNR_10log10'] = tb_summary['SNR_10log10'], tb_summary['SNR_10log10']
    return dict(toolbox=(tb_summary, tb_rows), paper=(paper_summary, paper_rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs-root', default='runs/exp')
    ap.add_argument('--out', default=None, help='default: <runs-root>/rescore')
    args = ap.parse_args()
    out = args.out or os.path.join(args.runs_root, 'rescore')
    paths = sorted(glob.glob(os.path.join(args.runs_root, '**', 'saved_test_outputs', '*_outputs.pickle'),
                             recursive=True))
    if not paths:
        raise SystemExit('No *_outputs.pickle under %s' % args.runs_root)
    os.makedirs(out, exist_ok=True)
    summary_rows, subject_rows = [], []
    for path in paths:
        run = os.path.relpath(path, args.runs_root).split(os.sep + 'saved_test_outputs')[0]
        results = score(path)
        subjects = ' '.join(r['subject'] for r in results['paper'][1])
        for protocol, (summary, rows) in results.items():
            summary_rows.append(dict(run=run, protocol=protocol, **{k: summary[k] for k in
                                     ('MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR', 'SNR_10log10', 'videos')},
                                     test_subjects=subjects, outputs=path))
            for row in rows:
                subject_rows.append(dict(run=run, protocol=protocol, **row))
    for name, rows in (('rescore_summary.csv', summary_rows), ('rescore_per_subject.csv', subject_rows)):
        with open(os.path.join(out, name), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    print('%-48s %-8s %6s %6s %6s %7s %7s' % ('run', 'protocol', 'MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR'))
    for r in summary_rows:
        print('%-48s %-8s %6.2f %6.2f %6.2f %7.4f %7.2f' % (r['run'][:48], r['protocol'], r['MAE'], r['RMSE'],
                                                          r['MAPE'], r['Pearson'], r['SNR']))
    splits = {r['test_subjects'] for r in summary_rows}
    if len(splits) > 1:
        print('\nWARNING: runs use %d different test-subject sets; compare only within a set:' % len(splits))
        for s in sorted(splits):
            print('  ', s)
    print('\nSaved:', os.path.join(out, 'rescore_summary.csv'), 'and rescore_per_subject.csv')


if __name__ == '__main__':
    main()
