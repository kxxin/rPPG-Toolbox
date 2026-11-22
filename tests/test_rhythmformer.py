"""RhythmFormer checks.  From the rPPG-Toolbox root:  python tests/test_rhythmformer.py   (CPU is fine: no Mamba)
1. ARCH rhythmformer = the toolbox RhythmFormer called on [B,T,C,H,W]; output [B,T]; wrong clip size is refused.
2. configs differ from the RhythmMamba baseline configs only in ARCH (+ LR for the _LR9E3 variants).
3. MT9 = MT6 teachers + 3 RhythmFormer seeds.
"""
import glob
import os
import sys

import torch
import yaml

sys.path.insert(0, os.getcwd())
C = 'configs/train_configs/'


def flat(d, p=''):
    out = {}
    for k, v in d.items():
        out.update(flat(v, p + k + '.') if isinstance(v, dict) else {p + k: v})
    return out


def test_model():
    from neural_methods.model.BiMambaPhys import BiMambaPhys
    from neural_methods.model.RhythmFormer import RhythmFormer
    torch.manual_seed(0)
    model = BiMambaPhys(dim=96, depth=24, arch='rhythmformer').eval()
    torch.manual_seed(0)
    ref = RhythmFormer().eval()
    ref.load_state_dict(model.reference.net.state_dict())
    video = torch.randn(1, 3, 160, 128, 128)
    with torch.no_grad():
        out = model(video)
        direct = ref(video.permute(0, 2, 1, 3, 4).contiguous())
    assert out.shape == (1, 160) and torch.allclose(out, direct, atol=1e-5)
    try:
        model(torch.randn(1, 3, 128, 128, 128))
        raise AssertionError('wrong clip length accepted')
    except ValueError:
        pass
    print('ok  rhythmformer adapter = toolbox RhythmFormer on [B,T,C,H,W]; %d params' % sum(p.numel() for p in model.parameters()))


def test_configs():
    files = sorted(glob.glob(C + '*_RHYTHMFORMER*_RM_SPEEDAUG_*.yaml'))
    assert len(files) == 8, files
    for path in files:
        twin = path.replace('_RHYTHMFORMER_LR9E3_', '_RHYTHMMAMBA_').replace('_RHYTHMFORMER_', '_RHYTHMMAMBA_')
        with open(path) as h:
            c = flat(yaml.safe_load(h))
        if os.path.isfile(twin):
            with open(twin) as h:
                t = flat(yaml.safe_load(h))
            diff = {k for k in set(c) | set(t) if c.get(k) != t.get(k)}
            allowed = {'MODEL.BIMAMBAPHYS.ARCH', 'TRAIN.MODEL_FILE_NAME', 'LOG.PATH'} | ({'TRAIN.LR'} if 'LR9E3' in path else set())
            assert diff == allowed, (path, diff)
        assert c['MODEL.BIMAMBAPHYS.ARCH'] == 'rhythmformer' and c['MODEL.BIMAMBAPHYS.LOSS.TYPE'] == 'rhythmmamba'
        assert abs(float(c['TRAIN.LR']) - (9e-3 if 'LR9E3' in path else 3e-4)) < 1e-12, path
    print('ok  8 configs: only ARCH (and LR for _LR9E3) differ from the RhythmMamba baselines')


def test_mt9():
    with open(C + 'UBFC_PURE_BIMAMBA_STUDENT_SINGLE_MT9_RM_SPEEDAUG_CROSS.yaml') as h:
        runs = yaml.safe_load(h)['DISTILL']['TEACHER_RUNS']
    assert len(runs) == 9 and sum('rhythmformer' in r for r in runs) == 3
    print('ok  MT9 = DPM x3 + RM x3 + RF x3')


if __name__ == '__main__':
    test_model()
    test_configs()
    test_mt9()
    print('all RhythmFormer tests passed')
