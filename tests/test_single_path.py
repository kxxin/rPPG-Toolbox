"""Checks for the single-path (NOISE_PATH=False) student.

Run from the rPPG-Toolbox root:  python tests/test_single_path.py          (CUDA, real Mamba)
Optional: --dual-checkpoint runs/exp/pinned_student6_nokd_2stage/experiment00/PreTrainedModels/
          UBFC_UBFC_BiMambaPhys_StudentNoKD2StagePinned_Epoch29.pth
          -> verifies the single-path model reproduces that checkpoint's noise-path-off output.
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from neural_methods.model.BiMambaPhys import BiMambaPhys
from neural_methods.loss.BiMambaPhysLoss import BiMambaPhysLoss

ROOT = Path(__file__).resolve().parents[1]


def count(model):
    return sum(p.numel() for p in model.parameters())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--dual-checkpoint', type=Path)
    args = ap.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required for real Mamba kernels')
    torch.manual_seed(0)
    single = BiMambaPhys(dim=64, depth=6, dual_path=True, noise_path=False).to(args.device)
    dual = BiMambaPhys(dim=64, depth=6, dual_path=True, noise_path=True).to(args.device)
    d = single.dual
    assert not hasattr(d, 'blocks_noise') and not hasattr(d, 'cross_attn') and len(d.blocks_rppg) == 6
    single_keys, dual_keys = set(single.state_dict()), set(dual.state_dict())
    assert single_keys < dual_keys
    assert all(k.startswith(('dual.blocks_noise.', 'dual.cross_attn.')) for k in dual_keys - single_keys)
    print('PASS: single path = dual minus noise blocks/cross-attention; params %d vs %d (%.1f %%)'
          % (count(single), count(dual), 100 * count(single) / count(dual)))

    single.train()
    x = torch.randn(2, 3, 160, 32, 32, device=args.device); y = torch.randn(2, 160, device=args.device)
    BiMambaPhysLoss()(single(x), y).backward()
    for part in (d.frame_stem, d.attention_conv, d.blocks_rppg, d.predictor):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in part.parameters())
    single.eval()
    with torch.no_grad():
        torch.testing.assert_close(single(x, stage=1), single(x, stage=2))
        out, aux = d(x, return_aux=True)
        assert out.shape == (2, 160) and 'modeled_noise' not in aux and aux['mask'].shape[1] == 1
    print('PASS: gradients reach stem/attention/rPPG path/predictor; stage argument is a no-op')

    # Weight compatibility: dual rPPG-path weights in the single model == dual stage-1 inference.
    if args.dual_checkpoint:
        state = torch.load(args.dual_checkpoint, map_location='cpu', weights_only=True)
        dual.load_state_dict(state, strict=True)
    missing, unexpected = single.load_state_dict(dual.state_dict(), strict=False)
    assert not missing and all(k.startswith(('dual.blocks_noise.', 'dual.cross_attn.')) for k in unexpected)
    dual.eval(); single.eval()
    with torch.no_grad():
        torch.testing.assert_close(single(x), dual(x, stage=1))
    print('PASS: single path with dual weights reproduces dual noise-path-off output%s'
          % (' (checkpoint)' if args.dual_checkpoint else ''))

    from config import _C
    for name in ('UBFC_UBFC_BIMAMBA_STUDENT_SINGLE_NOKD_PINNED.yaml', 'UBFC_UBFC_BIMAMBA_STUDENT_SINGLE_KD_PINNED.yaml'):
        cfg = _C.clone(); cfg.defrost(); cfg.merge_from_file(str(ROOT / 'configs' / 'train_configs' / name))
        m = cfg.MODEL.BIMAMBAPHYS
        assert m.DUAL_PATH and not m.NOISE_PATH and m.STAGE1_EPOCHS == 0 and m.DEPTH == 6
        assert cfg.TEST.DATA.FILE_LIST_PATH.endswith('UBFC-rPPG_intra_test.csv') and m.SPLIT_MANIFESTS
    print('PASS: single-path configs (pinned split, NOISE_PATH False, STAGE1_EPOCHS 0)')


if __name__ == '__main__':
    main()
