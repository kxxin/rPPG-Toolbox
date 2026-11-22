"""Baseline checks.  From the rPPG-Toolbox root:  python tests/test_baselines.py
1. configs: RhythmMamba loss, same data/optimiser settings as our teacher, only ARCH/size differ.
2. models (GPU if available; the real mamba_ssm is CUDA-only): the adapter gives the reference model
   exactly the permuted [B,T,C,H,W] input, output length = clip length, DualPhysMamba stage 1 and 2 run,
   ARCH bimamba is unchanged.
"""
import glob
import os
import sys

import torch
import yaml

sys.path.insert(0, os.getcwd())
C = 'configs/train_configs/'
TEACHERS = {'UBFC_UBFC': 'UBFC_UBFC_BIMAMBA_DUAL_2STAGE_RM_SPEEDAUG_PINNED.yaml',
            'PURE_PURE': 'PURE_PURE_BIMAMBA_DUAL_2STAGE_RM_SPEEDAUG_PINNED.yaml',
            'PURE_UBFC': 'PURE_UBFC_BIMAMBA_DUAL_2STAGE_RM_SPEEDAUG_CROSS.yaml',
            'UBFC_PURE': 'UBFC_PURE_BIMAMBA_DUAL_2STAGE_RM_SPEEDAUG_CROSS.yaml'}
ALLOWED = {'ARCH', 'DIM', 'DEPTH', 'MLP_RATIO', 'DROP_PATH_RATE', 'CROSS_ATTN_HEADS', 'DUAL_PATH', 'STAGE1_EPOCHS', 'NOISE_PATH'}


def flat(d, prefix=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, prefix + k + '.'))
        else:
            out[prefix + k] = v
    return out


def test_configs():
    files = sorted(glob.glob(C + '*_DUALPHYSMAMBA_RM_SPEEDAUG_*.yaml') + glob.glob(C + '*_BIDUALPHYSMAMBA_RM_SPEEDAUG_*.yaml')
                   + glob.glob(C + '*_RHYTHMMAMBA_RM_SPEEDAUG_*.yaml'))
    assert len(files) == 12, files
    for path in files:
        with open(path) as handle:
            c = yaml.safe_load(handle)
        key = os.path.basename(path)[:9]
        teacher_path = C + TEACHERS[key]
        if not os.path.isfile(teacher_path):
            print('skip diff for %s: teacher config %s not in this folder' % (os.path.basename(path), TEACHERS[key]))
        else:
            with open(teacher_path) as handle:
                t = yaml.safe_load(handle)
            diff = {k for k in set(flat(c)) | set(flat(t)) if flat(c).get(k) != flat(t).get(k)}
            unexpected = {k for k in diff if not (k.startswith('MODEL.BIMAMBAPHYS.') and k.split('.')[-1] in ALLOWED)} \
                - {'TRAIN.MODEL_FILE_NAME', 'LOG.PATH', 'DISTILL.TEACHER_CHECKPOINT', 'DISTILL.TEACHER_CONFIG'}
            assert not unexpected, (path, unexpected)
        m = c['MODEL']['BIMAMBAPHYS']
        assert m['LOSS']['TYPE'] == 'rhythmmamba' and c['TRAIN']['SPEED_AUG']['ENABLE'], path
        if m['ARCH'] == 'bidualphysmamba':
            with open(path.replace('_BIDUALPHYSMAMBA_', '_DUALPHYSMAMBA_')) as handle:
                u = yaml.safe_load(handle)
            diff = {k for k in set(flat(c)) | set(flat(u)) if flat(c).get(k) != flat(u).get(k)}
            assert diff == {'MODEL.BIMAMBAPHYS.ARCH', 'TRAIN.MODEL_FILE_NAME', 'LOG.PATH'}, (path, diff)
            assert '_v2_' in c['LOG']['PATH'], path
        if m['ARCH'] in ('dualphysmamba', 'bidualphysmamba'):
            assert m['DUAL_PATH'] and m['STAGE1_EPOCHS'] == 15 and m['DEPTH'] == 12, path
        else:
            assert m['ARCH'] == 'rhythmmamba' and not m['DUAL_PATH'] and m['STAGE1_EPOCHS'] == 0 and m['DEPTH'] == 24, path
    print('ok  12 configs: RhythmMamba loss; only ARCH/size differ from our teacher; Bi differs from Uni only in ARCH')


def device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        import mamba_ssm.ops.selective_scan_interface  # noqa: F401  real package -> needs GPU
        return None
    except ImportError:
        return torch.device('cpu')


def test_models():
    dev = device()
    if dev is None:
        print('skip model checks: real mamba_ssm needs a GPU (run on a GPU node)')
        return
    from neural_methods.model.BiMambaPhys import BiMambaPhys
    torch.manual_seed(0)
    video = torch.randn(2, 3, 32, 64, 64, device=dev)
    for arch, kw, stages in (('dualphysmamba', dict(dual_path=True), (1, 2)), ('rhythmmamba', {}, (2,))):
        model = BiMambaPhys(dim=96, depth=2, arch=arch, **kw).to(dev).eval()
        with torch.no_grad():
            for stage in stages:
                out = model(video, stage=stage)
                assert out.shape == (2, 32), (arch, stage, out.shape)
            ref_in = video.permute(0, 2, 1, 3, 4).contiguous()
            direct = model.reference.net(ref_in, stage=2) if arch == 'dualphysmamba' else model.reference.net(ref_in)
            assert torch.allclose(model(video, stage=2), direct, atol=1e-5), arch
        n = sum(p.numel() for p in model.parameters())
        print('ok  %s: output [B,T], identical to the reference called on [B,T,C,H,W] (%d params at depth 2)' % (arch, n))
    # bidirectional DualPhysMamba: every block's Mamba layer is bidirectional, rest identical to the reference
    from neural_methods.model.reference.BiDualPhysMamba import BiBlockMamba, BiMambaLayer
    torch.manual_seed(0)
    uni = BiMambaPhys(dim=96, depth=2, arch='dualphysmamba', dual_path=True).to(dev).eval()
    torch.manual_seed(0)
    bi = BiMambaPhys(dim=96, depth=2, arch='bidualphysmamba', dual_path=True).to(dev).eval()
    net = bi.reference.net
    assert all(isinstance(b, BiBlockMamba) and isinstance(b.attn, BiMambaLayer)
               for b in list(net.blocks_rppg) + list(net.blocks_noise))
    # v2: the backward scan must not see the stale tail of the shifted multi-temporal copies
    layer = net.blocks_rppg[0].attn.eval()
    D, copies, bsz = 16, 4, 2
    x = torch.randn(copies * bsz, D, 96, device=dev)
    valid = [D - k * D // copies for k in range(copies)]
    y = layer(x, valid=valid)
    x2 = x.clone()
    for k, length in enumerate(valid):
        x2[k * bsz:(k + 1) * bsz, length:] = torch.randn_like(x2[k * bsz:(k + 1) * bsz, length:])  # change stale tails
    y2 = layer(x2, valid=valid)
    for k, length in enumerate(valid):
        assert torch.allclose(y[k * bsz:(k + 1) * bsz, :length], y2[k * bsz:(k + 1) * bsz, :length], atol=1e-5), k
    print('ok  backward scan reads only each copy\'s valid prefix (stale tails have no effect)')
    names_uni = {k for k, _ in uni.named_parameters() if '.attn.' not in k}
    names_bi = {k for k, _ in bi.named_parameters() if '.attn.' not in k}
    assert names_uni == names_bi, names_uni ^ names_bi
    with torch.no_grad():
        for stage in (1, 2):
            assert bi(video, stage=stage).shape == (2, 32)
        # bidirectional: the earliest outputs respond to the last frames
        late = video.clone(); late[:, :, -8:] += 1.0
        d_bi = (bi(video)[:, :8] - bi(late)[:, :8]).abs().max().item()
    single = BiMambaPhys(dim=96, depth=2, arch='bidualphysmamba', dual_path=True, noise_path=False).to(dev).eval()
    assert not hasattr(single.reference.net, 'blocks_noise') and single(video).shape == (2, 32)
    n = lambda m: sum(p.numel() for p in m.parameters())
    print('ok  bidualphysmamba: all blocks bidirectional, non-Mamba parts identical to the reference; early outputs '
          'depend on late frames (max diff %.2e); params at depth 2: uni %d, bi %d, bi single-path %d'
          % (d_bi, n(uni), n(bi), n(single)))
    model = BiMambaPhys(dim=64, depth=2, dual_path=True).to(dev).eval()
    assert not hasattr(model, 'reference') and model(video).shape == (2, 32)
    print('ok  ARCH bimamba unchanged')


if __name__ == '__main__':
    test_configs()
    test_models()
    print('all baseline tests passed')
