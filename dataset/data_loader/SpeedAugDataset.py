"""Training-only speed (heart-rate) augmentation over an existing cached split.

Why: PURE's standard split trains on subjects 1-6 (45-90 bpm) and tests on subject 7
(125-135 bpm). A model that never sees high heart rates can lock onto a lower
frequency. Temporal resampling of a clip AND its label changes the pulse rate
consistently, so the training set can cover the whole HR range.

How (per training sample, with probability PROB):
  1. clip HR is estimated once at start-up from its cached label (FFT peak, 0.6-3.3 Hz);
  2. a target HR is drawn uniformly from [TARGET_HR_MIN, TARGET_HR_MAX] bpm;
  3. speed factor s = target / clip_HR, clipped to [MIN_FACTOR, MAX_FACTOR]
     (s > 1 speeds up = higher HR, s < 1 slows down = lower HR);
  4. 160 output frames are taken at positions start + s * [0..159] with linear
     interpolation in time; for s > 1 the window extends into the following cached
     clips of the SAME video (shifted back at the end of a video; s is reduced if the
     whole video is too short). Video and label get identical positions.
MODE 'balanced' (HR-balanced): the target HR is drawn first, then a training clip that can
reach it within [MIN_FACTOR, MAX_FACTOR] is used (the indexed clip whenever it can, otherwise
a random eligible clip), so augmented samples are ~uniform over the target range instead of
inheriting the dataset's HR skew (UBFC: median ~104 bpm, 0.5 % of clips < 55 bpm).
Without augmentation the original clip is returned unchanged. Dataset length, the
clip each index points to, and therefore steps per epoch are unchanged. Test and
validation loaders are never wrapped. Randomness uses numpy's global RNG, which
main.py's seed_worker seeds per worker.
"""
from collections import defaultdict
import json
import os

import numpy as np
from torch.utils.data import Dataset


def clip_hr_bpm(label, fs, low=0.6, high=3.3, nfft=4096):
    x = np.asarray(label, dtype=np.float64)
    x = (x - x.mean()) * np.hanning(len(x))
    f = np.fft.rfftfreq(nfft, 1 / fs)
    p = np.abs(np.fft.rfft(x, nfft)) ** 2
    band = (f >= low) & (f <= high)
    return float(f[band][np.argmax(p[band])] * 60)


class SpeedAugDataset(Dataset):
    def __init__(self, base, fs=30, clip_length=160, prob=0.5, target_hr=(40.0, 160.0),
                 factor_range=(0.5, 2.2), summary_path=None, mode='clip_target'):
        if base.data_format != 'NCDHW':
            raise ValueError('SpeedAugDataset requires DATA_FORMAT NCDHW')
        if mode not in ('clip_target', 'balanced'):
            raise ValueError('SPEED_AUG.MODE must be clip_target or balanced')
        self.mode = mode
        if not (0 <= prob <= 1 and 0 < target_hr[0] < target_hr[1] and 0 < factor_range[0] < 1 < factor_range[1]):
            raise ValueError('Invalid SPEED_AUG settings')
        self.base, self.inputs, self.data_format = base, list(base.inputs), base.data_format
        self.fs, self.T, self.prob = fs, clip_length, prob
        self.target_hr, self.factor_range = target_hr, factor_range
        videos = defaultdict(list)
        for index, path in enumerate(self.inputs):
            stem, chunk = os.path.basename(str(path).replace('\\', '/')).rsplit('_input', 1)
            videos[stem].append((int(chunk.split('.')[0]), index))
        self.position = {}  # dataset index -> (video, chunk number, ordered dataset indices of the video)
        for stem, pairs in videos.items():
            pairs.sort()
            if [c for c, _ in pairs] != list(range(len(pairs))):
                raise ValueError('Non-contiguous cached chunks for %s: speed augmentation needs whole videos' % stem)
            order = [i for _, i in pairs]
            for chunk, index in pairs:
                self.position[index] = (stem, chunk, order)
        self.clip_hr = np.array([clip_hr_bpm(np.load(base.labels[i]), fs) for i in range(len(self.inputs))])
        self.lo_reach = self.clip_hr * factor_range[0]
        self.hi_reach = self.clip_hr * factor_range[1]
        if summary_path:
            effective = self.simulate_effective_hr()
            edges = [0, 45, 55, 65, 80, 100, 120, 140, 1000]
            hist = np.histogram(effective, bins=edges)[0] / len(effective)
            per_video = {stem: [round(float(self.clip_hr[i]), 1) for _, i in sorted(pairs)]
                         for stem, pairs in videos.items()}
            summary = dict(prob=prob, target_hr_bpm=list(target_hr), factor_range=list(factor_range),
                           clips=len(self.inputs), videos=len(videos),
                           train_clip_hr_bpm=dict(min=float(self.clip_hr.min()), p5=float(np.percentile(self.clip_hr, 5)),
                                                  median=float(np.median(self.clip_hr)),
                                                  p95=float(np.percentile(self.clip_hr, 95)), max=float(self.clip_hr.max())),
                           reachable_hr_bpm=[float(self.clip_hr.min() * factor_range[0]),
                                             float(self.clip_hr.max() * factor_range[1])],
                           mode=mode,
                           effective_train_hr_share={'%d-%d' % (a, b): round(float(h), 4)
                                                     for a, b, h in zip(edges[:-1], edges[1:], hist)},
                           clip_hr_per_video=per_video)
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, 'w') as handle:
                json.dump(summary, handle, indent=2)
            print('Speed augmentation (%s): p=%.2f, target %g-%g bpm, factor %.2f-%.2f; train clip HR %.0f-%.0f bpm '
                  '(median %.0f); effective training HR < 55 bpm: %.1f %%'
                  % (mode, prob, target_hr[0], target_hr[1], factor_range[0], factor_range[1],
                     self.clip_hr.min(), self.clip_hr.max(), np.median(self.clip_hr),
                     100 * float(np.mean(effective < 55))), flush=True)

    def __len__(self):
        return len(self.inputs)

    def simulate_effective_hr(self, n=100000, seed=0):
        """Monte-Carlo HR distribution of training samples under the current settings (for the summary)."""
        state = np.random.get_state(); np.random.seed(seed)
        try:
            out = np.empty(n)
            for k in range(n):
                index = np.random.randint(len(self.inputs))
                if np.random.random_sample() >= self.prob:
                    out[k] = self.clip_hr[index]
                else:
                    source, s = self.plan(index)
                    out[k] = self.clip_hr[source] * s
            return out
        finally:
            np.random.set_state(state)

    def plan(self, index):
        """(source clip, speed factor) for an augmented sample of `index`."""
        if self.mode == 'clip_target':
            return index, self.factor_for(index)
        target = np.random.uniform(*self.target_hr)
        if self.lo_reach[index] <= target <= self.hi_reach[index]:
            source = index
        else:
            eligible = np.flatnonzero((self.lo_reach <= target) & (target <= self.hi_reach))
            if len(eligible) == 0:
                return index, self.factor_for(index)
            source = int(eligible[np.random.randint(len(eligible))])
        return source, float(target / max(self.clip_hr[source], 1e-6))

    def factor_for(self, index):
        target = np.random.uniform(*self.target_hr)
        return float(np.clip(target / max(self.clip_hr[index], 1e-6), *self.factor_range))

    def resample(self, index, s):
        """Returns (video [C,T,H,W], label [T]) resampled with speed factor s."""
        stem, chunk, order = self.position[index]
        total = len(order) * self.T
        span = s * (self.T - 1)
        if span > total - 1:
            s = (total - 1) / (self.T - 1); span = total - 1
        start = min(chunk * self.T, total - 1 - span)
        pos = start + s * np.arange(self.T)
        lo = np.floor(pos).astype(int); hi = np.minimum(lo + 1, total - 1); w = (pos - lo).astype(np.float32)
        first, last = lo[0] // self.T, hi[-1] // self.T
        items = [self.base[order[c]] for c in range(first, last + 1)]
        video = np.concatenate([it[0] for it in items], axis=1)
        label = np.concatenate([it[1] for it in items], axis=0)
        lo, hi = lo - first * self.T, hi - first * self.T
        out_video = video[:, lo] * (1 - w)[None, :, None, None] + video[:, hi] * w[None, :, None, None]
        out_label = label[lo] * (1 - w) + label[hi] * w
        return out_video.astype(np.float32), out_label.astype(np.float32), s

    def __getitem__(self, index):
        if np.random.random_sample() >= self.prob:
            return self.base[index]
        source, s = self.plan(index)
        stem, chunk, _ = self.position[source]  # same (filename, chunk_id) the base loader would return
        video, wave, _ = self.resample(source, s)
        return video, wave, stem, str(chunk)


def maybe_wrap(dataset, config):
    """Wrap the TRAIN dataset if TRAIN.SPEED_AUG.ENABLE; returns dataset unchanged otherwise."""
    aug = getattr(config.TRAIN, 'SPEED_AUG', None)
    if aug is None or not aug.ENABLE:
        return dataset
    if config.TRAIN.SEQUENCE_LENGTH:
        raise ValueError('SPEED_AUG and SEQUENCE_LENGTH>0 are separate experiments; enable only one')
    if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE != 'Standardized':
        raise ValueError('SPEED_AUG interpolates labels; it supports LABEL_TYPE Standardized only')
    return SpeedAugDataset(dataset, fs=config.TRAIN.DATA.FS, clip_length=config.TRAIN.DATA.PREPROCESS.CHUNK_LENGTH,
                           prob=aug.PROB, target_hr=(aug.TARGET_HR_MIN, aug.TARGET_HR_MAX),
                           factor_range=(aug.MIN_FACTOR, aug.MAX_FACTOR),
                           summary_path=os.path.join(config.MODEL.MODEL_DIR, 'speed_aug_summary.json'),
                           mode=getattr(aug, 'MODE', 'clip_target'))
