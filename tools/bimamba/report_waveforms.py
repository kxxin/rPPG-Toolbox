"""Unshifted waveform metrics and plots supplement toolbox HR metrics.

Pickles remain compatible with tools/output_signal_viz/data_out_viz.ipynb.
No filtering, lag search, or sign flip is used in the metrics below.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def normalized(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / max(x.std(), 1e-6)


def report_waveforms(predictions, labels, fs, output_dir, checkpoint=''):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for subject in sorted(predictions):
        chunks = sorted(predictions[subject])
        if chunks != list(range(len(chunks))):
            raise ValueError('Noncontiguous chunks for ' + subject)
        pp, yy = [], []
        for chunk in chunks:
            p, y = (np.asarray(source[subject][chunk]).reshape(-1) for source in (predictions, labels))
            if p.shape != y.shape or not np.isfinite(p).all() or not np.isfinite(y).all():
                raise ValueError('Invalid prediction/label for ' + subject)
            p, y = normalized(p), normalized(y)
            rows.append(dict(subject=subject, chunk=chunk, samples=len(p),
                             pearson=float(np.mean(p * y)), normalized_mae=float(np.mean(np.abs(p - y))),
                             normalized_rmse=float(np.sqrt(np.mean((p - y) ** 2)))))
            pp.append(p)
            yy.append(y)
        p, y = np.concatenate(pp), np.concatenate(yy)
        np.savez_compressed(output / (subject + '_waveforms.npz'),
                            time_seconds=np.arange(len(p)) / fs, prediction=p, ground_truth=y)
        # Same fixed first ten seconds for every subject, no best-clip selection.
        n = min(len(p), int(10 * fs))
        fig, axes = plt.subplots(2, 1, figsize=(11, 6))
        axes[0].plot(np.arange(n) / fs, y[:n], label='Contact BVP', linewidth=1.3)
        axes[0].plot(np.arange(n) / fs, p[:n], label='Estimated rPPG', alpha=0.8, linewidth=1.2)
        axes[0].set(xlabel='Time (s)', ylabel='Standardized amplitude',
                    title=subject + ' — first 10 s, unfiltered and unshifted')
        axes[0].legend()
        frequencies = np.fft.rfftfreq(len(p), 1 / fs)
        for signal, label in ((y, 'Contact BVP'), (p, 'Estimated rPPG')):
            power = np.abs(np.fft.rfft(signal * np.hanning(len(signal)))) ** 2
            axes[1].plot(frequencies, power / max(power.sum(), 1e-12), label=label)
        axes[1].set(xlim=(0.5, 4), xlabel='Frequency (Hz)', ylabel='Relative power', title='Full retained recording')
        axes[1].legend()
        fig.tight_layout()
        fig.savefig(output / (subject + '_comparison.png'), dpi=150)
        plt.close(fig)
    with (output / 'waveform_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metrics = ('pearson', 'normalized_mae', 'normalized_rmse')
    subjects = sorted(predictions)
    per_subject = {s: {k: float(np.mean([r[k] for r in rows if r['subject'] == s])) for k in metrics} for s in subjects}
    summary = dict(checkpoint=checkpoint, subjects=len(subjects), clips=len(rows),
                   protocol='Per-clip z-score; no bandpass, time shift, or polarity correction.',
                   macro_subject_mean={k: float(np.mean([v[k] for v in per_subject.values()])) for k in metrics},
                   per_subject=per_subject)
    (output / 'waveform_summary.json').write_text(json.dumps(summary, indent=2))
    print('Waveform metrics:', json.dumps(summary['macro_subject_mean']), flush=True)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pickle_path', help='Trusted local toolbox output pickle')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()
    with open(args.pickle_path, 'rb') as handle:
        result = pickle.load(handle)
    report_waveforms(result['predictions'], result['labels'], result['fs'], args.output_dir)
