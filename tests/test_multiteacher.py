"""Multi-teacher KD checks (CPU OK with a stand-in Mamba; real mamba_ssm needs a GPU for the model part).
From the rPPG-Toolbox root:  python tests/test_multiteacher.py
"""
import os
import sys

import torch
import yaml

sys.path.insert(0, os.getcwd())
from neural_methods.trainer.BiMambaPhysKDTrainer import TeacherEnsemble, _zscore  # noqa: E402

C = 'configs/train_configs/'


class Const(torch.nn.Module):
    def __init__(self, wave):
        super().__init__()
        self.wave = wave

    def forward(self, video, stage=2):
        return self.wave.expand(video.shape[0], -1)


def test_ensemble():
    t = torch.linspace(0, 6.28, 160)
    a, b = torch.sin(t) * 5 + 3, torch.sin(t + 0.3) * 0.1  # different scale/offset: z-scoring equalises them
    ens = TeacherEnsemble([Const(a), Const(b)])
    out = ens(torch.zeros(2, 3, 160, 8, 8))
    expect = _zscore(torch.stack([_zscore(a), _zscore(b)]).mean(0)).expand(2, -1)
    assert out.shape == (2, 160) and torch.allclose(out, expect, atol=1e-5)
    assert abs(out.mean().item()) < 1e-5 and abs(out.std(-1).mean().item() - 1) < 1e-3
    print('ok  ensemble = mean of z-scored teacher waveforms (scale-free), re-z-scored')


def test_configs():
    for v, n in (('MT6', 6), ('MT3DPM', 3)):
        with open(C + 'UBFC_PURE_BIMAMBA_STUDENT_SINGLE_%s_RM_SPEEDAUG_CROSS.yaml' % v) as handle:
            c = yaml.safe_load(handle)
        d, m = c['DISTILL'], c['MODEL']['BIMAMBAPHYS']
        assert len(d['TEACHER_RUNS']) == n and d['WEIGHT'] == 0.5 and not any(d['FEATURE'].values()), v
        assert m['LOSS']['TYPE'] == 'rhythmmamba' and m['NOISE_PATH'] is False and m['DEPTH'] == 6, v
        assert c['TEST']['USE_LAST_EPOCH'] is False and c['TRAIN']['SPEED_AUG']['ENABLE'], v
        assert all('cross_ubfc2pure_' in r for r in d['TEACHER_RUNS']), v
    print('ok  MT6 / MT3DPM configs: RhythmMamba loss, single-path 6-block student, output KD 0.5, best-valid epoch')


def test_run_seeds_no_pairing():
    sys.path.insert(0, os.path.join(os.getcwd(), 'tools', 'bimamba'))
    import run_seeds
    with open(C + 'UBFC_PURE_BIMAMBA_STUDENT_SINGLE_MT6_RM_SPEEDAUG_CROSS.yaml') as handle:
        c = yaml.safe_load(handle)
    assert run_seeds.is_kd(c) is False  # named teachers: no --teacher-config pairing
    print('ok  run_seeds treats multi-teacher configs as self-contained (no seed pairing)')


if __name__ == '__main__':
    test_ensemble()
    test_configs()
    test_run_seeds_no_pairing()
    print('all multi-teacher tests passed')
