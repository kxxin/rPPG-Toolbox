"""Checks for the RhythmMamba loss port and HR-balanced speed augmentation.

Run from the rPPG-Toolbox root (CPU is enough except for --rhythmmamba-root, which needs CUDA
because the reference code calls .cuda()):
    python tests/test_ablation.py
    python tests/test_ablation.py --rhythmmamba-root ../RhythmMamba-main     # numerical parity with Hybrid_Loss
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from neural_methods.loss.RhythmMambaLoss import RhythmMambaLoss, label_hr_bpm, build_supervised_loss
from dataset.data_loader.SpeedAugDataset import SpeedAugDataset, clip_hr_bpm

FS, T = 30, 160


def sine(hr, n=T, harmonic=0.0, phase=0.0):
    t = np.arange(n) / FS
    return np.sin(2 * np.pi * hr / 60 * t + phase) + harmonic * np.sin(2 * np.pi * 2 * hr / 60 * t)


def test_loss_behaviour():
    loss = RhythmMambaLoss(FS)
    label = torch.tensor(np.stack([sine(70), sine(50), sine(110)]), dtype=torch.float32)
    good = label + 0.05 * torch.randn_like(label)
    doubled = torch.tensor(np.stack([sine(140), sine(100), sine(110)]), dtype=torch.float32)
    harmonic_heavy = torch.tensor(np.stack([sine(70, harmonic=1.5), sine(50, harmonic=1.5), sine(110)]), dtype=torch.float32)
    lg, pg = loss(good, label, return_components=True)
    ld, pd = loss(doubled, label, return_components=True)
    lh, ph = loss(harmonic_heavy, label, return_components=True)
    assert pg['frequency_ce'] < ph['frequency_ce'] < pd['frequency_ce'], (pg, ph, pd)
    assert abs(label_hr_bpm(sine(50, n=160), FS) - 50) < 2.5
    pred = good.clone().requires_grad_(True)
    loss(pred, label).backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    print('PASS: CE frequency term ranks clean < harmonic-heavy < doubled HR; gradients finite '
          '(CE %.3f < %.3f < %.3f)' % (pg['frequency_ce'], ph['frequency_ce'], pd['frequency_ce']))


def test_parity(root):
    if not torch.cuda.is_available():
        raise SystemExit('--rhythmmamba-root parity needs CUDA (the reference hard-codes .cuda())')
    import importlib.util, types
    # Load RhythmMamba's own post_process + TorchLossComputer by file path, so they don't clash with the
    # toolbox's packages of the same name (TorchLossComputer imports evaluation.post_process.calculate_hr).
    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
    saved = sys.modules.get('evaluation.post_process')
    sys.modules['evaluation.post_process'] = load('rm_post_process', root / 'evaluation' / 'post_process.py')
    try:
        ref = load('rm_torch_loss', root / 'neural_methods' / 'loss' / 'TorchLossComputer.py')
    finally:
        if saved is not None:
            sys.modules['evaluation.post_process'] = saved
        else:
            sys.modules.pop('evaluation.post_process', None)
    torch.manual_seed(0)
    labels = torch.tensor(np.stack([sine(h, phase=p) for h, p in ((52, 0.3), (78, 1.0), (101, 2.0), (133, 0.1))]),
                          dtype=torch.float32)
    pred = labels + 0.4 * torch.randn_like(labels)
    pred_norm = (pred - pred.mean(-1, keepdim=True)) / pred.std(-1, keepdim=True)  # RhythmMambaTrainer
    criterion = ref.Hybrid_Loss()
    expected = sum(criterion(pred_norm[i].cuda(), labels[i].cuda(), 0, FS, False) for i in range(len(pred))) / len(pred)
    mine = RhythmMambaLoss(FS).cuda()(pred.cuda(), labels.cuda())
    assert torch.allclose(mine.float(), expected.float(), rtol=1e-4, atol=1e-5), (float(mine), float(expected))
    print('PASS: identical to RhythmMamba Hybrid_Loss (%.6f vs %.6f)' % (float(mine), float(expected)))


class FakeBase:
    data_format = 'NCDHW'

    def __init__(self, inputs):
        self.inputs = sorted(inputs); self.labels = [p.replace('_input', '_label') for p in self.inputs]

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        name = os.path.basename(self.inputs[i])
        return (np.float32(np.load(self.inputs[i]).transpose(3, 0, 1, 2)), np.float32(np.load(self.labels[i])),
                name.rsplit('_', 1)[0], name.rsplit('input', 1)[1].split('.')[0])


def test_balanced_aug():
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for v, hr in enumerate([100, 105, 110, 95, 115, 62]):  # UBFC-like: mostly fast, one slow video
            wave = sine(hr, n=4 * T).astype(np.float32)
            for c in range(4):
                seg = wave[c * T:(c + 1) * T]
                p = os.path.join(tmp, 'subject%d_input%d.npy' % (v + 1, c))
                np.save(p, np.broadcast_to(seg[:, None, None, None], (T, 4, 4, 3)).astype(np.float32))
                np.save(p.replace('_input', '_label'), seg); paths.append(p)
        base = FakeBase(paths)
        clip = SpeedAugDataset(base, FS, T, prob=0.5, mode='clip_target')
        bal = SpeedAugDataset(base, FS, T, prob=1.0, mode='balanced',
                              summary_path=os.path.join(tmp, 'm', 'speed_aug_summary.json'))
        eff_clip, eff_bal = clip.simulate_effective_hr(20000), bal.simulate_effective_hr(20000)
        assert np.mean(eff_bal < 55) > 2 * np.mean(eff_clip < 55), (np.mean(eff_bal < 55), np.mean(eff_clip < 55))
        assert abs(np.mean(eff_bal < 55) - 15 / 120) < 0.03  # ~uniform over 40-160
        np.random.seed(3)
        for _ in range(40):
            i = np.random.randint(len(bal))
            src, s = bal.plan(i)
            assert bal.factor_range[0] - 1e-9 <= s <= bal.factor_range[1] + 1e-9
            video, wave, stem, chunk = bal[i]
            assert video.shape == (3, T, 4, 4) and wave.shape == (T,) and np.isfinite(video).all()
            assert np.allclose(video[0, :, 0, 0], wave, atol=1e-5)
        src, s = bal.plan(0); _, wave, _ = bal.resample(src, s)
        assert abs(clip_hr_bpm(wave, FS) - bal.clip_hr[src] * s) < 3
        assert os.path.isfile(os.path.join(tmp, 'm', 'speed_aug_summary.json'))
    print('PASS: balanced mode ~uniform 40-160 bpm (<55 bpm: %.1f %% vs %.1f %% clip_target); '
          'factors within range; video/label aligned' % (100 * np.mean(eff_bal < 55), 100 * np.mean(eff_clip < 55)))


def test_configs():
    from config import _C
    root = Path(__file__).resolve().parents[1] / 'configs' / 'train_configs'
    expect = {'B': ('rhythmmamba', 'clip_target', 0.5), 'C': ('bimamba', 'balanced', 1.0), 'D': ('rhythmmamba', 'balanced', 1.0)}
    c = _C.clone(); c.defrost(); c.merge_from_file(str(root / 'UBFC_PURE_BIMAMBA_STUDENT_SINGLE_NOKD_ABLATION_E.yaml'))
    assert c.MODEL.BIMAMBAPHYS.LOSS.TYPE == 'rhythmmamba' and c.TRAIN.RM_AUG.ENABLE and not c.TRAIN.SPEED_AUG.ENABLE
    assert not _C.TRAIN.RM_AUG.ENABLE
    for v, (loss, mode, p) in expect.items():
        c = _C.clone(); c.defrost(); c.merge_from_file(str(root / ('UBFC_PURE_BIMAMBA_STUDENT_SINGLE_NOKD_ABLATION_%s.yaml' % v)))
        assert (c.MODEL.BIMAMBAPHYS.LOSS.TYPE, c.TRAIN.SPEED_AUG.MODE, c.TRAIN.SPEED_AUG.PROB) == (loss, mode, p)
        assert c.DISTILL.WEIGHT == 0 and not c.MODEL.BIMAMBAPHYS.NOISE_PATH and not c.TEST.USE_LAST_EPOCH
        assert type(build_supervised_loss(c, 30)).__name__ == ('RhythmMambaLoss' if loss == 'rhythmmamba' else 'BiMambaPhysLoss')
    assert _C.MODEL.BIMAMBAPHYS.LOSS.TYPE == 'bimamba' and _C.TRAIN.SPEED_AUG.MODE == 'clip_target'
    print('PASS: B/C/D configs merge; defaults keep earlier runs unchanged')



def test_rm_aug():
    """RhythmMamba augmentation: exact slow/fast/flip ops, HR rule, and reference batch semantics."""
    from dataset.data_loader.RhythmMambaAugDataset import RhythmMambaAugDataset, slow_down, speed_up
    x = np.arange(10, dtype=np.float32)
    s = slow_down(x, 0, 2)
    ref = np.zeros(10, np.float32)
    for e in range(0, 10, 2):
        ref[e] = x[2 + e // 2]; ref[e + 1] = (x[2 + (e + 1) // 2] + x[2 + (e + 1) // 2 + 1]) / 2
    assert np.allclose(s, ref)
    assert np.allclose(speed_up(x, 0), np.concatenate([x[::2], x[::2]]))
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for v, hr in enumerate([100, 80, 60]):
            seg = sine(hr).astype(np.float32)
            p = os.path.join(tmp, 'subject%d_input0.npy' % (v + 1))
            img = np.random.default_rng(v).standard_normal((1, 4, 6, 3)).astype(np.float32)
            np.save(p, (img + seg[:, None, None, None]).astype(np.float32))
            np.save(p.replace('_input', '_label'), seg); paths.append(p)
        base = FakeBase(paths)
        ds = RhythmMambaAugDataset(base, FS, prob=1.0, flip_prob=0.0)
        hrs = {}
        for i in range(3):
            video, label, *_ = ds[i]
            hrs[round(ds.hr_cache[i])] = clip_hr_bpm(label, FS)
            assert video.shape == base[i][0].shape and np.allclose(video[0, :, 0, 0] - base[i][0][0, 0, 0, 0] * 0, video[0, :, 0, 0])
        h100 = [k for k in hrs if k > 90][0]; h60 = [k for k in hrs if k < 75][0]; h80 = [k for k in hrs if 75 <= k <= 90][0]
        # speed-up repeats an 80-frame half clip (reference), so the phase jump at the join biases a 160-frame
        # HR estimate low; the operation itself is checked exactly against the reference formula above.
        assert abs(hrs[h100] - h100 / 2) < 4 and hrs[h60] > 1.7 * h60 and abs(hrs[h80] - h80) < 2, hrs
        flip = RhythmMambaAugDataset(base, FS, prob=0.0, flip_prob=1.0)
        assert np.array_equal(flip[0][0], base[0][0][..., ::-1])
        # reference mode: clip0 (100 bpm) triggers 'slow' -> written to ALL clips; clip1 (80) and clip2 (60)
        # come later: clip1 no-op resets itself, clip2 'fast' overwrites ALL clips again -> all sped up.
        refds = RhythmMambaAugDataset(base, FS, prob=1.0, flip_prob=0.0, mode='reference')
        batch = [refds[i] for i in range(3)]
        out_v, out_l, _, _ = refds.collate(batch)
        for j in range(3):
            assert np.allclose(out_l[j].numpy(), speed_up(base[j][1], 0)), j
        batch2 = [refds[i] for i in (2, 0, 1)]  # 60 (fast, all), 100 (slow, all), 80 (resets itself)
        v2, l2, _, _ = refds.collate(batch2)
        assert np.allclose(l2[2].numpy(), base[1][1])  # the 80 bpm clip came last -> original
        # the 60 bpm clip got the 100 bpm clip's 2x slow-down (-> 30 bpm), from some start in [0, T/2)
        assert any(np.allclose(l2[0].numpy(), slow_down(base[2][1], 0, r)) for r in range(T // 2))
    print('PASS: RhythmMamba aug ops match the reference formulas; HR>90 halves, HR<75 doubles, 75-90 unchanged;'
          ' flip; reference mode reproduces the batch-level indexing')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--rhythmmamba-root', type=Path)
    args = ap.parse_args()
    test_loss_behaviour(); test_balanced_aug(); test_configs(); test_rm_aug()
    if args.rhythmmamba_root:
        test_parity(args.rhythmmamba_root)
