"""Checks for the two-stage, spatial-mask BiMambaPhys dual model.

Run from the rPPG-Toolbox root after copying the files in:
    python tests/test_bimamba_dual_2stage.py
    python tests/test_bimamba_dual_2stage.py --old-checkpoint runs/exp/bimamba_dual12/experiment00/PreTrainedModels/UBFC_UBFC_BiMambaPhys_Dual_Epoch29.pth
Real Mamba CUDA kernels are required for the model checks (--device cuda).
"""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from neural_methods.model.BiMambaPhys import BiMambaPhys
from neural_methods.model.BiMambaPhysDual import attention_masks
from neural_methods.loss.BiMambaPhysLoss import BiMambaPhysLoss
from neural_methods.trainer.BiMambaPhysTrainer import BiMambaPhysTrainer

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ['UBFC_UBFC_BIMAMBA_DUAL_2STAGE.yaml', 'UBFC_UBFC_BIMAMBA_STUDENT_NOKD_2STAGE.yaml',
           'UBFC_UBFC_BIMAMBA_KD_2STAGE.yaml']


def has_grad(module):
    return any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())


def no_grad(module):
    return all(p.grad is None for p in module.parameters())


def test_masks():
    logits = torch.randn(2, 1, 7, 5, 6) * 3
    mask, inverse = attention_masks(logits)
    assert mask.shape == inverse.shape == logits.shape
    torch.testing.assert_close(mask.mean((-2, -1)), torch.full((2, 1, 7), .5))
    torch.testing.assert_close(inverse.mean((-2, -1)), torch.full((2, 1, 7), .5))
    assert (mask >= 0).all() and (inverse >= 0).all()
    # Complement is taken before normalization: high-attention pixels get low inverse.
    order = logits.flatten(-2).argsort(-1)
    m, i = mask.flatten(-2).gather(-1, order), inverse.flatten(-2).gather(-1, order)
    assert (m[..., 1:] >= m[..., :-1] - 1e-6).all() and (i[..., 1:] <= i[..., :-1] + 1e-6).all()
    flat = attention_masks(torch.zeros(1, 1, 3, 4, 4))
    torch.testing.assert_close(flat[0], torch.full((1, 1, 3, 4, 4), .5))
    torch.testing.assert_close(flat[1], flat[0])
    for value in (100., -100.):
        out = attention_masks(torch.full((1, 1, 2, 3, 3), value))
        assert all(torch.isfinite(x).all() for x in out)
    try:
        attention_masks(torch.zeros(1, 4, 2, 3, 3))
        raise AssertionError('multi-channel logits must be rejected')
    except ValueError:
        pass
    print('PASS: single-channel spatial masks, mean 0.5/frame, complement-before-normalize')


def test_schedule_and_configs():
    trainer = SimpleNamespace(stage1_epochs=15)
    stages = [BiMambaPhysTrainer._stage(trainer, e) for e in range(30)]
    assert stages == [1] * 15 + [2] * 15
    assert [BiMambaPhysTrainer._stage(SimpleNamespace(stage1_epochs=0), e) for e in range(3)] == [2, 2, 2]
    from config import _C
    assert _C.MODEL.BIMAMBAPHYS.STAGE1_EPOCHS == 0, 'config.py default must keep old configs single-stage'
    for name in CONFIGS:
        cfg = _C.clone(); cfg.defrost()
        cfg.merge_from_file(str(ROOT / 'configs' / 'train_configs' / name))
        m = cfg.MODEL.BIMAMBAPHYS
        assert m.DUAL_PATH and 0 < m.STAGE1_EPOCHS < cfg.TRAIN.EPOCHS, name
    print('PASS: stage schedule 15x stage1 + 15x stage2; configs merge with config.py')


def test_model(device, old_checkpoint):
    torch.manual_seed(100)
    model = BiMambaPhys(dim=64, depth=12, dual_path=True).to(device).train()
    d = model.dual
    assert d.attention_conv.out_channels == 1
    if old_checkpoint:
        try:
            model.load_state_dict(torch.load(old_checkpoint, map_location='cpu', weights_only=True), strict=True)
            raise AssertionError('old per-channel-mask checkpoint unexpectedly loaded')
        except RuntimeError as error:
            assert 'attention_conv' in str(error)
        print('PASS: old dual checkpoint is rejected (missing attention_conv) -> retrain teacher')
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)
    loss_fn = BiMambaPhysLoss()
    x = torch.randn(2, 3, 160, 32, 32, device=device)
    y = torch.randn(2, 160, device=device)

    # Stage 1: rPPG path, stem, attention conv and predictor learn; noise path untouched.
    noise_before = [p.detach().clone() for p in list(d.blocks_noise.parameters()) + list(d.cross_attn.parameters())]
    optimizer.zero_grad(set_to_none=True)
    loss_fn(model(x, stage=1), y).backward()
    for part in (d.frame_stem, d.stem3, d.attention_conv, d.blocks_rppg, d.predictor):
        assert has_grad(part), part.__class__.__name__
    assert no_grad(d.blocks_noise) and no_grad(d.cross_attn)
    optimizer.step()
    noise_after = list(d.blocks_noise.parameters()) + list(d.cross_attn.parameters())
    assert all(torch.equal(a, b) for a, b in zip(noise_before, noise_after)), 'AdamW touched noise path in stage 1'
    print('PASS: stage 1 trains rPPG path only; noise path/cross-attention unchanged by AdamW')

    # Stage 2: everything learns, including the attention conv through BOTH masks.
    optimizer.zero_grad(set_to_none=True)
    loss_fn(model(x, stage=2), y).backward()
    for part in (d.frame_stem, d.attention_conv, d.blocks_rppg, d.blocks_noise, d.cross_attn, d.predictor):
        assert has_grad(part), part.__class__.__name__
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step()
    print('PASS: stage 2 joint gradients incl. noise path, cross-attention, attention conv')

    model.eval()
    with torch.no_grad():
        for t in (159, 160, 161):
            v = torch.randn(1, 3, t, 32, 32, device=device)
            for stage in (1, 2):
                out, aux = d(v, stage=stage, return_aux=True)
                assert out.shape == (1, t) and torch.isfinite(out).all()
                assert aux['mask'].shape[1] == 1 and aux['inverse_mask'].shape[1] == 1
            torch.testing.assert_close(aux['clean'], aux['rppg'] - aux['modeled_noise'])
            torch.testing.assert_close(model(v), d(v, stage=2))  # default = stage 2
    print('PASS: output lengths 159/160/161 for both stages; default forward is stage 2')
    print('parameters:', sum(p.numel() for p in model.parameters()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--old-checkpoint', type=Path)
    args = ap.parse_args()
    test_masks()
    test_schedule_and_configs()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required for the real Mamba scan checks')
    test_model(args.device, args.old_checkpoint)


if __name__ == '__main__':
    main()
