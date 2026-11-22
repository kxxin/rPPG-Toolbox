"""RhythmMamba / DualPhysMamba training loss: 0.2 * NegPearson + 1.0 * frequency cross-entropy.

Faithful, device-agnostic port of RhythmMamba-main/neural_methods/loss/TorchLossComputer.py
(Hybrid_Loss, used by RhythmMambaTrainer and the reproduced DualPhysMambaTrainer):

  per clip i in the batch (the reference loops over samples and averages):
    pred_i  <- (pred_i - mean) / std                       (RhythmMambaTrainer normalises before the loss)
    time_i  =  1 - Pearson(pred_i, label_i)
    hr_gt_i =  Welch peak of the LABEL after detrend(100) + 1st-order Butterworth 0.75-2.5 Hz
               (RhythmMamba evaluation/post_process.calculate_hr; computed on CPU, no gradient)
    power_i =  |DFT(pred_i * hann)|^2 at every integer bpm 45..149   (105 'classes')
    share_i =  power_i / sum(power_i)          (reference complex_absolute: 'analogous softmax')
    ce_i    =  cross_entropy(logits=share_i, class=int(hr_gt_i - 45))
  loss = mean_i (0.2 * time_i + 1.0 * ce_i)

As in the reference, the CE logits are the band-normalised power shares (0..1, summing to 1),
so CE lies between log(105)-1 ~ 3.65 (all power in the true bin) and ~log(105) ~ 4.65;
the term is shallow but always pushes power toward the true fundamental bin, and power at the
2nd harmonic counts as a wrong class.
The reference's unused Gaussian-label KL term is not computed.
"""
import math

import numpy as np
import scipy.signal
import torch
from scipy.signal import butter, welch
from scipy.sparse import spdiags
from torch import nn
from torch.nn import functional as F

BPM_LOW, BPM_HIGH = 45, 150  # torch.arange(45, 150) in the reference


def _detrend(signal, lambda_value=100):
    n = signal.shape[0]
    H = np.identity(n)
    D = spdiags(np.array([np.ones(n), -2 * np.ones(n), np.ones(n)]), np.array([0, 1, 2]), n - 2, n).toarray()
    return np.dot(H - np.linalg.inv(H + (lambda_value ** 2) * np.dot(D.T, D)), signal)


def label_hr_bpm(label, fs):
    """RhythmMamba calculate_hr(...)[1] for diff_flag=False."""
    x = _detrend(np.asarray(label, dtype=np.float64), 100)
    b, a = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    x = scipy.signal.filtfilt(b, a, x)
    f, p = welch(x, fs, nfft=1e5 / fs, nperseg=np.min((len(x) - 1, 256)))
    band = (f > 45 / 60) & (f < 150 / 60)
    return float(f[band][np.argmax(p[band])] * 60)


class RhythmMambaLoss(nn.Module):
    def __init__(self, fs=30.0, time_weight=0.2, frequency_weight=1.0):
        super().__init__()
        self.fs, self.time_weight, self.frequency_weight = float(fs), float(time_weight), float(frequency_weight)
        self.register_buffer('bpm', torch.arange(BPM_LOW, BPM_HIGH, dtype=torch.float32), persistent=False)

    def spectrum(self, x):
        """[N] -> power at each integer bpm 45..149 (reference compute_complex_absolute_given_k)."""
        n = x.shape[-1]
        k = (self.bpm.to(x.device) / 60.0) / (self.fs / n)
        phase = 2 * math.pi * torch.arange(n, dtype=torch.float32, device=x.device) / n
        windowed = x.float() * torch.from_numpy(np.hanning(n)).float().to(x.device)
        arg = k.view(-1, 1) * phase.view(1, -1)
        return (windowed.view(1, -1) * torch.sin(arg)).sum(-1) ** 2 + (windowed.view(1, -1) * torch.cos(arg)).sum(-1) ** 2

    def forward(self, prediction, target, return_components=False):
        if prediction.ndim != 2 or prediction.shape != target.shape:
            raise ValueError('Prediction and target must be [B,T] with matching shapes')
        times, ces = [], []
        for pred, label in zip(prediction, target):
            pred = pred.float()
            std = pred.std()
            pred = (pred - pred.mean()) / std.clamp_min(1e-8)
            y = label.float()
            px, py = pred - pred.mean(), y - y.mean()
            pearson = (px * py).sum() / torch.sqrt((px * px).sum() * (py * py).sum()).clamp_min(1e-12)
            times.append(1 - pearson)
            hr_gt = label_hr_bpm(label.detach().cpu().numpy(), self.fs)
            cls = torch.tensor([int(hr_gt - BPM_LOW)], device=pred.device, dtype=torch.long)
            power = self.spectrum(pred)
            ces.append(F.cross_entropy((power / power.sum()).view(1, -1), cls))
        time_loss, ce_loss = torch.stack(times).mean(), torch.stack(ces).mean()
        total = self.time_weight * time_loss + self.frequency_weight * ce_loss
        if return_components:
            return total, {'pearson': time_loss, 'frequency_ce': ce_loss}
        return total


class CombinedLoss(nn.Module):
    """BiMamba loss (Pearson + L1 + spectral JS) + RhythmMamba frequency cross-entropy.

    total = PEARSON_WEIGHT*pearson + WAVE_WEIGHT*L1 + FREQUENCY_WEIGHT*JS + RM_FREQUENCY_WEIGHT*CE
    The RhythmMamba time term is not added again (the BiMamba Pearson term already covers it).
    Motivation: the BiMamba loss won intra-dataset and PURE->UBFC; the RhythmMamba CE (a classification
    of the label HR bin, so power at 2xHR is penalised) is what reduced octave errors on UBFC->PURE.
    """

    def __init__(self, bimamba, rhythmmamba_ce_weight):
        super().__init__()
        self.bimamba = bimamba
        self.ce = RhythmMambaLoss(bimamba.fs, time_weight=0.0, frequency_weight=1.0)
        self.ce_weight = float(rhythmmamba_ce_weight)

    def forward(self, prediction, target, return_components=False):
        base, parts = self.bimamba(prediction, target, return_components=True)
        _, rm_parts = self.ce(prediction, target, return_components=True)
        total = base + self.ce_weight * rm_parts['frequency_ce']
        if return_components:
            return total, dict(parts, frequency_ce=rm_parts['frequency_ce'])
        return total


def build_supervised_loss(config, fs):
    """MODEL.BIMAMBAPHYS.LOSS.TYPE: 'bimamba' (default, Pearson+L1+JS), 'rhythmmamba', or 'combined'
    (BiMamba loss + RM_FREQUENCY_WEIGHT x RhythmMamba frequency CE)."""
    loss = config.MODEL.BIMAMBAPHYS.LOSS
    kind = getattr(loss, 'TYPE', 'bimamba')
    if kind == 'combined':
        from neural_methods.loss.BiMambaPhysLoss import BiMambaPhysLoss
        bimamba = BiMambaPhysLoss(fs, loss.PEARSON_WEIGHT, loss.WAVE_WEIGHT, loss.FREQUENCY_WEIGHT, loss.MIN_HZ, loss.MAX_HZ)
        return CombinedLoss(bimamba, getattr(loss, 'RM_FREQUENCY_WEIGHT', 1.0))
    if kind == 'rhythmmamba':
        return RhythmMambaLoss(fs, getattr(loss, 'RM_TIME_WEIGHT', 0.2), getattr(loss, 'RM_FREQUENCY_WEIGHT', 1.0))
    if kind == 'bimamba':
        from neural_methods.loss.BiMambaPhysLoss import BiMambaPhysLoss
        return BiMambaPhysLoss(fs, loss.PEARSON_WEIGHT, loss.WAVE_WEIGHT, loss.FREQUENCY_WEIGHT, loss.MIN_HZ, loss.MAX_HZ)
    raise ValueError('Unknown MODEL.BIMAMBAPHYS.LOSS.TYPE: %s' % kind)
