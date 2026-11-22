"""Per-clip Pearson + standardized waveform L1 + in-band spectral JS loss."""
import torch
from torch import nn


def normalize_waveform(x, eps=1e-6):
    x = x - x.mean(dim=-1, keepdim=True)
    return x / x.square().mean(dim=-1, keepdim=True).clamp_min(eps ** 2).sqrt()


class BiMambaPhysLoss(nn.Module):
    def __init__(self, fs=30.0, pearson_weight=1.0, wave_weight=0.5,
                 frequency_weight=0.1, min_hz=0.7, max_hz=3.0, eps=1e-6):
        super().__init__()
        if not 0 < min_hz < max_hz < fs / 2:
            raise ValueError("Frequency band must be inside (0, Nyquist).")
        if min(pearson_weight, wave_weight, frequency_weight) < 0:
            raise ValueError("Loss weights must be nonnegative.")
        if pearson_weight + wave_weight + frequency_weight == 0:
            raise ValueError("At least one loss weight must be positive.")
        self.fs, self.min_hz, self.max_hz, self.eps = fs, min_hz, max_hz, eps
        self.weights = (pearson_weight, wave_weight, frequency_weight)

    def forward(self, prediction, target, return_components=False):
        if prediction.ndim != 2 or prediction.shape != target.shape or prediction.shape[-1] < 4:
            raise ValueError("Prediction and target must have matching [B,T] shapes, T >= 4.")
        p, y = normalize_waveform(prediction.float()), normalize_waveform(target.float())
        pearson = (1 - (p * y).mean(dim=-1).clamp(-1, 1)).mean()
        wave = (p - y).abs().mean()
        frequencies = torch.fft.rfftfreq(p.shape[-1], d=1 / self.fs, device=p.device)
        band = (frequencies >= self.min_hz) & (frequencies <= self.max_hz)
        if band.sum() < 2:
            raise ValueError("Clip too short to resolve at least two physiological frequency bins.")
        window = torch.hann_window(p.shape[-1], device=p.device)
        pp = torch.fft.rfft(p * window, dim=-1).abs().square()[..., band] + self.eps
        yy = torch.fft.rfft(y * window, dim=-1).abs().square()[..., band] + self.eps
        pp, yy = pp / pp.sum(-1, keepdim=True), yy / yy.sum(-1, keepdim=True)
        mixture = (pp + yy) / 2
        frequency = (0.5 * ((pp * (pp.log() - mixture.log())).sum(-1)
                           + (yy * (yy.log() - mixture.log())).sum(-1))).mean()
        total = sum(w * v for w, v in zip(self.weights, (pearson, wave, frequency)))
        if return_components:
            return total, {"pearson": pearson, "wave": wave, "frequency": frequency}
        return total
