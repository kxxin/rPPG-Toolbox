"""CPU checks for SpeedAugDataset. Run from the rPPG-Toolbox root: python tests/test_speed_aug.py
Optional real-cache check (no training):
    python tests/test_speed_aug.py --file-list /home/user/24020695/PreprocessedData/splits/PURE_intra_train.csv
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from dataset.data_loader.SpeedAugDataset import SpeedAugDataset, clip_hr_bpm, maybe_wrap

FS, T = 30, 160


class FakeBase:
    """Mimics BaseLoader: inputs/labels lists, NCDHW __getitem__ -> (data, label, name, chunk)."""
    data_format = 'NCDHW'

    def __init__(self, inputs):
        self.inputs = sorted(inputs)
        self.labels = [p.replace('_input', '_label') for p in self.inputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        data = np.float32(np.load(self.inputs[i]).transpose(3, 0, 1, 2))
        name = os.path.basename(self.inputs[i])
        return data, np.float32(np.load(self.labels[i])), name.rsplit('_', 1)[0], name.rsplit('input', 1)[1].split('.')[0]


def make_video(folder, stem, hr_bpm, clips):
    t = np.arange(clips * T) / FS
    wave = np.sin(2 * np.pi * hr_bpm / 60 * t).astype(np.float32)
    paths = []
    for c in range(clips):
        seg = wave[c * T:(c + 1) * T]
        video = np.broadcast_to(seg[:, None, None, None], (T, 4, 4, 3)).astype(np.float32)
        p = os.path.join(folder, '%s_input%d.npy' % (stem, c))
        np.save(p, video); np.save(p.replace('_input', '_label'), seg); paths.append(p)
    return paths


def test_synthetic():
    with tempfile.TemporaryDirectory() as tmp:
        paths = make_video(tmp, '101', 60.0, 6) + make_video(tmp, '201', 85.0, 4) + make_video(tmp, '301', 70.0, 1)
        base = FakeBase(paths)
        ds = SpeedAugDataset(base, FS, T, prob=1.0, summary_path=os.path.join(tmp, 'm', 'speed_aug_summary.json'))
        assert ds.inputs == base.inputs and len(ds) == len(base)
        assert os.path.isfile(os.path.join(tmp, 'm', 'speed_aug_summary.json'))
        hr = {os.path.basename(p).split('_')[0]: ds.clip_hr[i] for i, p in enumerate(ds.inputs)}
        assert abs(hr['101'] - 60) < 1.5 and abs(hr['201'] - 85) < 1.5, hr
        i101 = [i for i, p in enumerate(ds.inputs) if '/101_' in p]
        # s = 1 reproduces the clip exactly
        v, y, s = ds.resample(i101[2], 1.0)
        d0, l0, *_ = base[i101[2]]
        assert s == 1.0 and np.array_equal(v, d0) and np.array_equal(y, l0)
        # speed-up / slow-down change HR by the factor, video and label consistently
        for s_req, idx in ((2.0, i101[0]), (2.0, i101[-1]), (0.6, i101[3]), (1.45, i101[4])):
            v, y, s = ds.resample(idx, s_req)
            assert v.shape == (3, T, 4, 4) and y.shape == (T,) and v.dtype == np.float32
            assert abs(clip_hr_bpm(y, FS) - 60 * s) < 2.5, (s, clip_hr_bpm(y, FS))
            assert np.allclose(v[0, :, 0, 0], y, atol=1e-5)
        # single-clip video: the factor is reduced to fit, never reads outside the video
        i301 = [i for i, p in enumerate(ds.inputs) if '/301_' in p][0]
        v, y, s = ds.resample(i301, 2.0)
        assert s == 1.0 and np.array_equal(y, base[i301][1])
        # target sampling covers the high-HR range that the clips lack
        np.random.seed(0)
        out = [clip_hr_bpm(ds[i][1], FS) for _ in range(30) for i in i101]
        assert max(out) > 120 and min(out) < 45, (min(out), max(out))
        # prob 0 = untouched
        same = SpeedAugDataset(base, FS, T, prob=0.0)
        for i in range(len(base)):
            assert np.array_equal(same[i][0], base[i][0])
        # non-contiguous chunks rejected
        try:
            SpeedAugDataset(FakeBase([p for p in paths if not p.endswith('101_input2.npy')]), FS, T)
            raise AssertionError('gap not rejected')
        except ValueError:
            pass
    print('PASS: exact identity at s=1; HR scales by s for video and label; end-of-video and short-video handling;'
          ' target sampling reaches >120 bpm from 60 bpm clips; prob=0 untouched; gaps rejected')


def test_config():
    from config import _C
    cfg = _C.clone(); cfg.defrost()
    base = SimpleNamespace(data_format='NCDHW', inputs=[], labels=[])
    assert maybe_wrap(base, cfg) is base
    cfg.TRAIN.SPEED_AUG.ENABLE = True; cfg.TRAIN.SEQUENCE_LENGTH = 640
    try:
        maybe_wrap(base, cfg); raise AssertionError('SEQUENCE_LENGTH + SPEED_AUG not rejected')
    except ValueError:
        pass
    root = Path(__file__).resolve().parents[1] / 'configs' / 'train_configs'
    for name in ('PURE_PURE_BIMAMBA_STUDENT_SINGLE_NOKD_SPEEDAUG_PINNED.yaml',
                 'UBFC_UBFC_BIMAMBA_STUDENT_SINGLE_NOKD_SPEEDAUG_PINNED.yaml'):
        c = _C.clone(); c.defrost(); c.merge_from_file(str(root / name))
        assert c.TRAIN.SPEED_AUG.ENABLE and not c.MODEL.BIMAMBAPHYS.NOISE_PATH and c.DISTILL.WEIGHT == 0
    print('PASS: off by default; refuses SEQUENCE_LENGTH; speed-aug configs merge')


def test_real(file_list):
    import pandas as pd
    inputs = pd.read_csv(file_list)['input_files'].tolist()
    base = FakeBase(inputs)
    ds = SpeedAugDataset(base, FS, T, prob=1.0)
    np.random.seed(1)
    picks = np.random.choice(len(ds), size=min(20, len(ds)), replace=False)
    got = []
    for i in picks:
        v, y, s = ds.resample(int(i), ds.factor_for(int(i)))
        assert np.isfinite(v).all() and np.isfinite(y).all() and v.shape[1] == T
        got.append((ds.clip_hr[i], s, clip_hr_bpm(y, FS)))
    print('clip HR -> factor -> augmented label HR (bpm):')
    for a, s, b in got:
        print('  %6.1f x %.2f -> %6.1f (expected %.1f)' % (a, s, b, a * s))
    print('PASS: real cache resamples cleanly; train clip HR range %.0f-%.0f bpm' % (ds.clip_hr.min(), ds.clip_hr.max()))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--file-list')
    args = ap.parse_args()
    test_synthetic(); test_config()
    if args.file_list:
        test_real(args.file_list)
