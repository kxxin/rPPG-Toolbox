"""RhythmMamba's training augmentation (TRAIN.AUG: 1 in its cross-dataset configs), as a loader option.

Source: RhythmMamba-main/neural_methods/trainer/RhythmMambaTrainer.py, data_augmentation(). For each
training clip (T frames), with probability 0.5:
  * clip HR > 90 bpm  -> 2x SLOW-DOWN: start r ~ U{0..T/2-1}; out[2i] = x[r+i],
                         out[2i+1] = (x[r+i] + x[r+i+1]) / 2      (HR halves, e.g. 104 -> 52 bpm)
  * clip HR < 75 bpm  -> 2x SPEED-UP: out[:T/2] = x[::2]; out[T/2:] = out[:T/2]  (HR doubles; the
                         half-length clip is repeated, as in the reference)
  * otherwise         -> unchanged
then, with probability 0.5, a horizontal flip (W axis). Video and label get identical operations.
Clip HR = RhythmMamba calculate_hr on the clip's label (detrend, 0.75-2.5 Hz Butterworth, Welch peak
in 45-150 bpm), cached per clip like the reference's data_dict.

MODE
  'per_sample' (default): the rules above applied to each clip independently -- the evident intent.
  'reference': reproduces the reference's batch-level indexing exactly (applied in collate, per batch):
       the reference writes data_aug[:, ...] (ALL clips) whenever ONE clip triggers a resample, and
       data_aug[idx] = data[idx] otherwise, looping idx = 0..N-1; the flip decision of the LAST clip
       applies to the whole batch. Net effect: every clip up to the last triggering clip receives THAT
       clip's resampling regardless of its own HR. Depends on batch size (reference: 16).
Training loader only; validation/test are never wrapped.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, default_collate

from neural_methods.loss.RhythmMambaLoss import label_hr_bpm


def slow_down(x, axis, start):
    """2x slow-down along `axis` starting at `start` (reference even/odd interpolation)."""
    t = x.shape[axis]
    half = np.arange(t) // 2
    base = np.take(x, start + half, axis=axis)
    nxt = np.take(x, np.minimum(start + half + 1, start + t // 2), axis=axis)
    odd = (np.arange(t) % 2 == 1)
    shape = [1] * x.ndim; shape[axis] = t
    odd = odd.reshape(shape)
    return np.where(odd, (base + nxt) / 2, base)


def speed_up(x, axis):
    t = x.shape[axis]
    first = np.take(x, np.arange(0, t, 2)[: t // 2], axis=axis)
    return np.concatenate([first, first], axis=axis)


class RhythmMambaAugDataset(Dataset):
    def __init__(self, base, fs=30, prob=0.5, flip_prob=0.5, hi_bpm=90.0, lo_bpm=75.0, mode='per_sample'):
        if base.data_format != 'NCDHW':
            raise ValueError('RhythmMambaAugDataset requires DATA_FORMAT NCDHW')
        if mode not in ('per_sample', 'reference'):
            raise ValueError('RM_AUG.MODE must be per_sample or reference')
        self.base, self.inputs, self.data_format = base, list(base.inputs), base.data_format
        self.labels = list(getattr(base, 'labels', []))
        self.fs, self.prob, self.flip_prob, self.hi, self.lo, self.mode = fs, prob, flip_prob, hi_bpm, lo_bpm, mode
        self.hr_cache = {}
        print('RhythmMamba augmentation (%s): p=%.2f slow-down if HR>%g, speed-up if HR<%g; flip p=%.2f'
              % (mode, prob, hi_bpm, lo_bpm, flip_prob), flush=True)

    def __len__(self):
        return len(self.base)

    def clip_hr(self, index, label):
        if index not in self.hr_cache:
            self.hr_cache[index] = label_hr_bpm(label, self.fs)
        return self.hr_cache[index]

    def decide(self, index, label):
        """'slow' | 'fast' | None for one clip (the reference's rand1 + HR rule)."""
        if np.random.random_sample() >= self.prob:
            return None
        hr = self.clip_hr(index, label)
        return 'slow' if hr > self.hi else ('fast' if hr < self.lo else None)

    @staticmethod
    def apply(video, label, op, start):
        """video [C,T,H,W], label [T]."""
        if op == 'slow':
            return slow_down(video, 1, start).astype(np.float32), slow_down(label, 0, start).astype(np.float32)
        if op == 'fast':
            return speed_up(video, 1).astype(np.float32), speed_up(label, 0).astype(np.float32)
        return video, label

    def __getitem__(self, index):
        video, label, name, chunk = self.base[index]
        if self.mode == 'reference':
            return video, label, name, chunk, index  # batch-level logic runs in collate
        t = video.shape[1]
        op = self.decide(index, label)
        video, label = self.apply(video, label, op, np.random.randint(0, t // 2))
        if np.random.random_sample() < self.flip_prob:
            video = video[..., ::-1].copy()
        return video, label, name, chunk

    def collate(self, batch):
        """Reference-mode batch augmentation (exact port of the indexing in data_augmentation)."""
        if self.mode != 'reference':
            return default_collate(batch)
        videos = np.stack([b[0] for b in batch]); labels = np.stack([b[1] for b in batch])
        n, t = labels.shape
        out_v, out_l = np.zeros_like(videos), np.zeros_like(labels)
        flip = False
        for idx, item in enumerate(batch):
            rand1, rand2 = np.random.random_sample(), np.random.random_sample()
            op = None
            if rand1 < self.prob:
                hr = self.clip_hr(item[4], labels[idx])
                op = 'slow' if hr > self.hi else ('fast' if hr < self.lo else None)
            if op is None:
                out_v[idx], out_l[idx] = videos[idx], labels[idx]
            else:  # reference writes ALL clips of the batch here
                start = np.random.randint(0, t // 2)
                for j in range(n):
                    out_v[j], out_l[j] = self.apply(videos[j], labels[j], op, start)
            flip = rand2 < self.flip_prob  # reference uses the last clip's rand2 for the whole batch
        if flip:
            out_v = out_v[..., ::-1].copy()
        return (torch.from_numpy(out_v), torch.from_numpy(out_l),
                [b[2] for b in batch], [b[3] for b in batch])


def maybe_wrap(dataset, config):
    aug = getattr(config.TRAIN, 'RM_AUG', None)
    if aug is None or not aug.ENABLE:
        return dataset, None
    if config.TRAIN.SEQUENCE_LENGTH:
        raise ValueError('RM_AUG and SEQUENCE_LENGTH>0 are separate experiments; enable only one')
    if config.TRAIN.DATA.PREPROCESS.LABEL_TYPE != 'Standardized':
        raise ValueError('RM_AUG supports LABEL_TYPE Standardized only (as used in your configs)')
    wrapped = RhythmMambaAugDataset(dataset, fs=config.TRAIN.DATA.FS, prob=aug.PROB, flip_prob=aug.FLIP_PROB,
                                    hi_bpm=aug.HI_BPM, lo_bpm=aug.LO_BPM, mode=aug.MODE)
    return wrapped, (wrapped.collate if aug.MODE == 'reference' else None)
