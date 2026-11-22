"""Run with python tests/test_bimamba_dual.py; real Mamba CUDA kernels required."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from neural_methods.model.BiMambaPhys import BiMambaPhys
from neural_methods.model.BiMambaPhysDual import FrequencyDomainFFN, attention_masks
from neural_methods.loss.BiMambaPhysLoss import BiMambaPhysLoss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', type=Path)
    ap.add_argument('--baseline-checkpoint', type=Path)
    ap.add_argument('--real-cache', type=Path)
    ap.add_argument('--train-size', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=2)
    args = ap.parse_args()
    torch.manual_seed(100)
    torch.set_num_threads(4)
    if args.reference:
        spec = importlib.util.spec_from_file_location('reference_model', args.reference)
        reference = importlib.util.module_from_spec(spec); spec.loader.exec_module(reference)
        old = reference.Frequencydomain_FFN(8, 2).eval()
        new = FrequencyDomainFFN(8, 2).eval(); new.load_state_dict(old.state_dict())
        x = torch.randn(2, 17, 8)
        torch.testing.assert_close(new(x), old(x))
        print('PASS: FDF numerical parity with supplied reference')
    for value in (0., 100., -100.):
        signal, noise = attention_masks(torch.full((2, 8, 11, 3, 3), value))
        assert torch.isfinite(signal).all() and torch.isfinite(noise).all()
        if value == 0.:
            torch.testing.assert_close(signal, torch.full_like(signal, .5))
            torch.testing.assert_close(noise, signal)
    if args.baseline_checkpoint:
        baseline = BiMambaPhys()
        baseline.load_state_dict(torch.load(args.baseline_checkpoint, map_location='cpu'), strict=True)
        del baseline
        print('PASS: legacy baseline checkpoint loads strictly')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required to test actual Mamba scan forward/backward')
    model = BiMambaPhys(dim=64, depth=12, dual_path=True).cuda().train()
    assert len(model.dual.blocks_rppg) == len(model.dual.blocks_noise) == 12
    assert not ({id(p) for p in model.dual.blocks_rppg.parameters()} &
                {id(p) for p in model.dual.blocks_noise.parameters()})
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    inputs = torch.randn(args.batch_size, 3, 160, args.train_size, args.train_size, device='cuda')
    labels = torch.randn(args.batch_size, 160, device='cuda')
    prediction = model(inputs)
    assert prediction.shape == labels.shape and torch.isfinite(prediction).all()
    loss = BiMambaPhysLoss()(prediction, labels); loss.backward()
    for path in (model.dual.blocks_rppg, model.dual.blocks_noise):
        for block in path:
            for branch in (block.attn.forward_mamba, block.attn.backward_mamba):
                grad = branch.in_proj.weight.grad
                assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
            assert block.mlp.r.grad is not None and block.mlp.r.grad.abs().sum() > 0
    for component in (model.dual.frame_stem, model.dual.cross_attn, model.dual.predictor):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in component.parameters())
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    print('PASS: 12+12 blocks, both directions, FDF, stem, cross-attention gradients + optimizer step')
    model.eval()
    with torch.no_grad():
        for t in (159, 160, 161):
            out, aux = model.dual(torch.randn(1,3,t,32,32,device='cuda'), return_aux=True)
            assert out.shape == (1,t) and torch.isfinite(out).all()
            torch.testing.assert_close(aux['clean'], aux['rppg']-aux['modeled_noise'])
        if args.real_cache:
            import numpy as np
            x = np.load(args.real_cache).astype(np.float32).transpose(3,0,1,2).copy()
            out = model(torch.from_numpy(x).unsqueeze(0).cuda())
            assert out.shape == (1,x.shape[1]) and torch.isfinite(out).all()
    print('PASS: exact output lengths 159/160/161, subtraction, and optional real cache')
    print(json.dumps(dict(parameters=sum(p.numel() for p in model.parameters()),
                          peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                          loss=float(loss.detach()), device=torch.cuda.get_device_name()), indent=2))


if __name__ == '__main__': main()
