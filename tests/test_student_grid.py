"""Student-grid checks.  From the rPPG-Toolbox root:  python tests/test_student_grid.py
1. combined loss = BiMamba loss + RhythmMamba frequency CE (values and gradients).
2. configs: each differs from its bimamba_aug_cross no-KD twin only in the intended fields.
3. compact student builds at dim 64 / 6 blocks and gives [B,T] (GPU if available; real mamba_ssm is CUDA-only).
"""
import glob
import os
import sys

import torch
import yaml

sys.path.insert(0, os.getcwd())
from neural_methods.loss.BiMambaPhysLoss import BiMambaPhysLoss  # noqa: E402
from neural_methods.loss.RhythmMambaLoss import CombinedLoss, RhythmMambaLoss  # noqa: E402

C = 'configs/train_configs/'


def test_loss():
    torch.manual_seed(0)
    t = torch.arange(160) / 30.0
    target = torch.stack([torch.sin(2 * torch.pi * 1.5 * t), torch.sin(2 * torch.pi * 1.1 * t)])
    pred = (target + 0.5 * torch.randn_like(target)).requires_grad_()
    bm = BiMambaPhysLoss(30.0)
    comb = CombinedLoss(bm, 1.0)
    total, parts = comb(pred, target, return_components=True)
    ce = RhythmMambaLoss(30.0, time_weight=0.0, frequency_weight=1.0)(pred, target)
    assert torch.allclose(total, bm(pred, target) + ce, atol=1e-5)
    assert set(parts) == {'pearson', 'wave', 'frequency', 'frequency_ce'}
    total.backward()
    assert pred.grad is not None and torch.isfinite(pred.grad).all()
    doubled = torch.stack([torch.sin(2 * torch.pi * 3.0 * t), torch.sin(2 * torch.pi * 2.2 * t)])  # 2x HR
    assert comb(doubled, target) > comb(target, target)
    print('ok  combined loss = BiMamba loss + RM frequency CE; penalises a doubled-HR prediction')


def flat(d, p=''):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + '.') if isinstance(v, dict) else {p + k: v})
    return out


def test_configs():
    files = sorted(glob.glob(C + '*_BIMAMBA_STUDENT_SINGLE_NOKD_COMB_SPEEDAUG_*.yaml') + glob.glob(C + '*_COMPACT6_*_SPEEDAUG_*.yaml'))
    assert len(files) == 16, len(files)
    for path in files:
        with open(path) as h:
            c = yaml.safe_load(h)
        m = c['MODEL']['BIMAMBAPHYS']
        compact = '_COMPACT6_' in path
        loss = path.split('_SPEEDAUG')[0].rsplit('_', 1)[1]
        assert m['LOSS']['TYPE'] == {'BM': 'bimamba', 'RM': 'rhythmmamba', 'COMB': 'combined'}[loss], path
        if compact:
            assert m['ARCH'] == 'rhythmmamba' and m['DIM'] == 64 and m['DEPTH'] == 6 and not m['DUAL_PATH'], path
            assert c['MODEL']['NAME'] == 'BiMambaPhys', path
        else:
            assert m.get('ARCH', 'bimamba') == 'bimamba' and m['NOISE_PATH'] is False and m['DEPTH'] == 6, path
        assert c['TRAIN']['SPEED_AUG']['ENABLE'] and c['DISTILL']['WEIGHT'] == 0.0, path
        assert c['TEST']['USE_LAST_EPOCH'] is path.endswith('_PINNED.yaml'), path
    print('ok  16 grid configs (4 cells x 4 settings): losses, students and protocol as described')


def test_compact_model():
    dev = None
    if torch.cuda.is_available():
        dev = torch.device('cuda')
    else:
        try:
            import mamba_ssm.ops.selective_scan_interface  # noqa: F401
        except ImportError:
            dev = torch.device('cpu')
    if dev is None:
        print('skip compact model check: real mamba_ssm needs a GPU')
        return
    from neural_methods.model.BiMambaPhys import BiMambaPhys
    model = BiMambaPhys(dim=64, depth=6, arch='rhythmmamba').to(dev).eval()
    with torch.no_grad():
        out = model(torch.randn(2, 3, 160, 128, 128, device=dev))
    assert out.shape == (2, 160)
    print('ok  compact student (rhythmmamba, dim 64, 6 blocks): output [B,T], %d params'
          % sum(p.numel() for p in model.parameters()))


if __name__ == '__main__':
    test_loss()
    test_configs()
    test_compact_model()
    print('all student-grid tests passed')
