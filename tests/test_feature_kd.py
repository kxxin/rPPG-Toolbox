"""Feature-KD checks (CPU OK; with the stub or real mamba_ssm).  From the toolbox root:
    python tests/test_feature_kd.py
1. return_aux exposes one rPPG feature per block, and the last equals the normal forward path.
2. block mapping 6 <- 12 is teacher blocks 2,4,...,12 (1-based).
3. each loss term is ~0 for identical aux and > 0 otherwise; gradients reach the student only.
4. the shipped configs have output KD off and the intended feature weights.
"""
import os
import sys

import torch
import yaml

sys.path.insert(0, os.getcwd())
from neural_methods.loss.FeatureDistillationLoss import FeatureDistillationLoss, block_mapping  # noqa: E402
from neural_methods.model.BiMambaPhysDual import DualPathBiMambaPhys  # noqa: E402


def build(depth, noise_path, dim=32):
    return DualPathBiMambaPhys(dim=dim, depth=depth, noise_path=noise_path).eval()


def mamba_device():
    """Real mamba_ssm kernels are CUDA-only; the CPU stand-in (tests here) has no selective_scan module."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        import mamba_ssm.ops.selective_scan_interface  # noqa: F401  (real package present)
        return None
    except ImportError:
        return torch.device('cpu')


def test_aux_blocks():
    device = mamba_device()
    if device is None:
        print('skip rppg_blocks check: real mamba_ssm needs a GPU (run on a GPU node)')
        return
    torch.manual_seed(0)
    video = torch.randn(1, 3, 32, 64, 64, device=device)
    for noise in (False, True):
        m = build(4, noise).to(device)
        with torch.no_grad():
            plain = m(video, stage=2)
            out, aux = m(video, stage=2, return_aux=True)
        assert len(aux['rppg_blocks']) == len(m.blocks_rppg)
        assert torch.allclose(aux['rppg_blocks'][-1], aux['rppg'], atol=1e-6)
        assert torch.allclose(out, plain, atol=1e-4), (out - plain).abs().max()
    print('ok  rppg_blocks match the Sequential forward (single and dual path, %s)' % device.type)


def test_mapping():
    assert block_mapping(6, 12) == [1, 3, 5, 7, 9, 11]
    assert block_mapping(12, 12) == list(range(12))
    print('ok  block mapping 6<-12 = teacher blocks 2,4,...,12')


def fake_aux(seed, blocks=3, requires_grad=False):
    g = torch.Generator().manual_seed(seed)
    aux = dict(mask=torch.rand(2, 1, 8, 4, 4, generator=g), rppg=torch.randn(2, 8, 16, generator=g),
               rppg_blocks=[torch.randn(2, 8, 16, generator=g) for _ in range(blocks)])
    if requires_grad:
        aux = dict(mask=aux['mask'].requires_grad_(), rppg=aux['rppg'].requires_grad_(),
                   rppg_blocks=[b.requires_grad_() for b in aux['rppg_blocks']])
    return aux


def test_terms():
    teacher = fake_aux(1, blocks=6)
    teacher['clean'] = teacher['rppg']
    same = dict(mask=teacher['mask'] * 3.0, rppg=teacher['clean'] * 2.0, rppg_blocks=teacher['rppg_blocks'][1::2])
    for name in ('at', 'hint', 'final', 'tsm'):
        loss = FeatureDistillationLoss(**{name + '_weight': 1.0})
        zero, _ = loss(same, teacher)  # scale-invariant terms -> 0
        assert float(zero) < 1e-5, (name, float(zero))
        student = fake_aux(2, blocks=3, requires_grad=True)
        val, parts = loss(student, teacher)
        assert float(val.detach()) > 1e-3, name
        val.backward()
    teacher = fake_aux(1, blocks=6, requires_grad=True)
    student = fake_aux(2, blocks=3, requires_grad=True)
    total, _ = FeatureDistillationLoss(1, 1, 1, 1)(student, teacher)
    total.backward()
    assert teacher['rppg'].grad is None and teacher['mask'].grad is None
    assert student['rppg'].grad is not None and student['mask'].grad is not None
    assert not FeatureDistillationLoss().active
    print('ok  AT/hint/final/TSM: 0 when matched (scale-invariant), >0 otherwise, no teacher gradient')


def test_configs():
    want = {'AT': (1, 0, 0, 0), 'HINT': (0, 1, 1, 0), 'ATHINT': (1, 1, 1, 0)}
    for v, w in want.items():
        path = 'configs/train_configs/PURE_PURE_BIMAMBA_STUDENT_SINGLE_FKD_%s_SPEEDAUG_PINNED.yaml' % v
        with open(path) as handle:
            c = yaml.safe_load(handle)
        f = c['DISTILL']['FEATURE']
        assert c['DISTILL']['WEIGHT'] == 0.0
        assert (f['AT_WEIGHT'], f['HINT_WEIGHT'], f['FINAL_WEIGHT'], f['TSM_WEIGHT']) == w
        assert c['MODEL']['BIMAMBAPHYS']['NOISE_PATH'] is False
        assert c['MODEL']['BIMAMBAPHYS']['LOSS']['TYPE'] == 'rhythmmamba'
    with open('configs/train_configs/PURE_PURE_BIMAMBA_STUDENT_SINGLE_NOKD_RM_SPEEDAUG_PINNED.yaml') as handle:
        c = yaml.safe_load(handle)
    assert c['MODEL']['BIMAMBAPHYS']['LOSS']['TYPE'] == 'rhythmmamba' and c['DISTILL']['WEIGHT'] == 0.0
    assert not any(c['DISTILL'].get('FEATURE', {}).values())
    teacher = 'runs/exp/pure_pinned_dual12_2stage_rm_speedaug/'
    for name in ('DUAL_2STAGE_RM', 'STUDENT_SINGLE_OUTKD_RM'):
        with open('configs/train_configs/PURE_PURE_BIMAMBA_%s_SPEEDAUG_PINNED.yaml' % name) as handle:
            c = yaml.safe_load(handle)
        assert c['MODEL']['BIMAMBAPHYS']['LOSS']['TYPE'] == 'rhythmmamba', name
    assert c['DISTILL']['WEIGHT'] > 0 and c['DISTILL']['TEACHER_CHECKPOINT'].startswith(teacher)
    for v in want:
        with open('configs/train_configs/PURE_PURE_BIMAMBA_STUDENT_SINGLE_FKD_%s_SPEEDAUG_PINNED.yaml' % v) as handle:
            assert yaml.safe_load(handle)['DISTILL']['TEACHER_CHECKPOINT'].startswith(teacher), v
    print('ok  configs: RhythmMamba loss everywhere, KD students -> RM teacher, feature-KD output KD off, feature weights as documented')


if __name__ == '__main__':
    test_mapping()
    test_terms()
    test_aux_blocks()
    test_configs()
    print('all feature-KD tests passed')
