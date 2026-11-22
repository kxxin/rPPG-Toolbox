"""Paper evaluation protocol (RhythmMamba / DualPhysMamba), alongside the toolbox one.

Numerically identical to RhythmMamba-main/evaluation/post_process.py +
metrics.py with EVALUATION_METHOD='FFT', USE_SMALLER_WINDOW=False:

  * detrend (lambda=100) -> 1st-order Butterworth band-pass 0.75-2.5 Hz (filtfilt)
  * HR = Welch PSD peak in (45, 150) bpm, nperseg=min(len-1, 256), nfft=1e5/fs
  * SNR vs. the label HR: power within +/-6 bpm of the 1st and 2nd harmonics over
    the remaining 0.75-2.5 Hz power (periodogram, nfft = next power of 2),
    converted with 20*log10 (RhythmMamba's mag2db) -- this is the convention of
    the published SNR tables. The current rPPG-Toolbox uses 10*log10; both are
    reported so the two are never mixed up.
  * MAE / RMSE / MAPE / Pearson over per-video HRs; SNR = mean per video.

Differences from the toolbox protocol (evaluation/post_process.py): toolbox uses a
whole-video periodogram (0.88 bpm bins at 1920 frames), 0.6-3.3 Hz band and
10*log10 SNR.
"""
import csv
import json
import os

import numpy as np
import scipy.signal
from scipy.signal import butter, welch
from scipy.sparse import spdiags

LOW_HZ, HIGH_HZ = 0.75, 2.5


def _detrend(signal, lambda_value=100):
    n = signal.shape[0]
    H = np.identity(n)
    D = spdiags(np.array([np.ones(n), -2 * np.ones(n), np.ones(n)]), np.array([0, 1, 2]), n - 2, n).toarray()
    return np.dot(H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D)), signal)


def welch_hr(y, fs=30, min_bpm=45, max_bpm=150):
    f, p = welch(y, fs, nfft=1e5 / fs, nperseg=np.min((len(y) - 1, 256)))
    band = (f > min_bpm / 60) & (f < max_bpm / 60)
    return f[band][np.argmax(p[band])] * 60


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def snr_ratio(pred, hr_label, fs=30, low=LOW_HZ, high=HIGH_HZ):
    """Harmonic-to-remainder power ratio (linear). Returns None if no remainder power."""
    f1, f2, dev = hr_label / 60, 2 * hr_label / 60, 6 / 60
    f, p = scipy.signal.periodogram(np.expand_dims(pred, 0), fs=fs,
                                    nfft=_next_power_of_2(len(pred)), detrend=False)
    p = np.squeeze(p)
    h1 = (f >= f1 - dev) & (f <= f1 + dev)
    h2 = (f >= f2 - dev) & (f <= f2 + dev)
    rem = (f >= low) & (f <= high) & ~h1 & ~h2
    if p[rem].sum() == 0:
        return None
    return (p[h1].sum() + p[h2].sum()) / p[rem].sum()


def metric_per_video(pred, label, fs=30, diff_flag=False):
    """Returns (hr_label, hr_pred, snr_20log10, snr_10log10)."""
    pred, label = np.asarray(pred, dtype=np.float64), np.asarray(label, dtype=np.float64)
    if diff_flag:
        pred, label = _detrend(np.cumsum(pred)), _detrend(np.cumsum(label))
    else:
        pred, label = _detrend(pred), _detrend(label)
    b, a = butter(1, [LOW_HZ / fs * 2, HIGH_HZ / fs * 2], btype='bandpass')
    pred = scipy.signal.filtfilt(b, a, pred)
    label = scipy.signal.filtfilt(b, a, label)
    hr_pred, hr_label = welch_hr(pred, fs), welch_hr(label, fs)
    ratio = snr_ratio(pred, hr_label, fs)
    snr20 = 0.0 if ratio is None else 20 * np.log10(ratio)
    snr10 = 0.0 if ratio is None else 10 * np.log10(ratio)
    return hr_label, hr_pred, snr20, snr10


def reform(chunks):
    """{chunk_id: tensor/array} -> 1-D array in sorted chunk order (as the toolbox)."""
    out = []
    for key in sorted(chunks):
        value = chunks[key]
        if hasattr(value, 'detach'):
            value = value.detach().cpu().numpy()
        out.append(np.asarray(value).reshape(-1))
    return np.concatenate(out)


def subject_sort_key(name):
    digits = ''.join(c for c in str(name) if c.isdigit())
    return (int(digits) if digits else 0, str(name))


def summarize(rows):
    gt = np.array([r['hr_label'] for r in rows]); pr = np.array([r['hr_pred'] for r in rows])
    err = pr - gt
    return dict(MAE=float(np.mean(np.abs(err))),
                RMSE=float(np.sqrt(np.mean(err ** 2))),
                MAPE=float(np.mean(np.abs(err) / gt) * 100),
                Pearson=float(np.corrcoef(pr, gt)[0, 1]) if len(rows) > 2 else float('nan'),
                SNR=float(np.mean([r['snr_db20'] for r in rows])),
                SNR_10log10=float(np.mean([r['snr_db10'] for r in rows])),
                videos=len(rows))


def evaluate(predictions, labels, fs=30, label_type='Standardized'):
    if label_type in ('Standardized', 'Raw'):
        diff_flag = False
    elif label_type == 'DiffNormalized':
        diff_flag = True
    else:
        raise ValueError('Unsupported label type: %s' % label_type)
    rows = []
    for subject in sorted(predictions, key=subject_sort_key):
        hr_label, hr_pred, snr20, snr10 = metric_per_video(reform(predictions[subject]),
                                                           reform(labels[subject]), fs, diff_flag)
        rows.append(dict(subject=str(subject), hr_label=float(hr_label), hr_pred=float(hr_pred),
                         error=float(hr_pred - hr_label), snr_db20=float(snr20), snr_db10=float(snr10)))
    return summarize(rows), rows


def report(predictions, labels, config, output_dir):
    """Print paper-protocol metrics and save json + per-subject csv next to the outputs."""
    summary, rows = evaluate(predictions, labels, config.TEST.DATA.FS, config.TEST.DATA.PREPROCESS.LABEL_TYPE)
    print('\n==== Paper protocol (RhythmMamba/DualPhysMamba: Welch HR, 0.75-2.5 Hz, SNR 20log10) ====')
    for key in ('MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR'):
        print('PAPER %s: %.4f%s' % (key, summary[key], ' (dB)' if key == 'SNR' else ''))
    print('PAPER SNR in 10log10 convention: %.4f (dB)' % summary['SNR_10log10'])
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'paper_protocol_metrics.json'), 'w') as handle:
        json.dump(dict(protocol='RhythmMamba/DualPhysMamba video-level Welch HR, 0.75-2.5 Hz, '
                                'SNR=20log10 harmonic/remainder power', **summary), handle, indent=2)
    with open(os.path.join(output_dir, 'paper_protocol_per_subject.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return summary, rows
