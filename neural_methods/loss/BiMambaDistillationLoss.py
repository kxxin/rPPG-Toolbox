"""Waveform-output KD; retain ground-truth supervision throughout training."""
import math
from torch import nn
from neural_methods.loss.BiMambaPhysLoss import BiMambaPhysLoss


class BiMambaDistillationLoss(nn.Module):
    def __init__(self, fs=30., pearson_weight=1., wave_weight=.5,
                 frequency_weight=.1, min_hz=.7, max_hz=3., kd_weight=.5, supervised=None):
        super().__init__()
        if not math.isfinite(kd_weight) or kd_weight < 0:
            raise ValueError('DISTILL.WEIGHT must be finite and nonnegative')
        self.kd_weight = kd_weight
        self.waveform = BiMambaPhysLoss(fs, pearson_weight, wave_weight,
                                      frequency_weight, min_hz, max_hz)
        # Ground-truth term: the trainer's LOSS.TYPE loss if given; the KD term stays the waveform loss.
        self.supervised = supervised if supervised is not None else self.waveform

    def forward(self, student, target, teacher=None):
        supervised, gt_parts = self.supervised(student, target, return_components=True)
        zero = supervised.new_zeros(())
        kd, kd_parts = zero, {'pearson': zero, 'wave': zero, 'frequency': zero}
        if teacher is not None and self.kd_weight > 0:
            # Detach here as well as no_grad at inference: the loss cannot train a teacher.
            kd, kd_parts = self.waveform(student, teacher.detach(), return_components=True)
        total = supervised + self.kd_weight * kd
        return total, dict(supervised=supervised, **gt_parts, kd=kd,
                           **{'kd_' + name: value for name, value in kd_parts.items()})
