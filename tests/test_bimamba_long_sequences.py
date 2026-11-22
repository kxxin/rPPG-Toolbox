"""Verify temporal joins, exposure matching and optional real-Mamba CUDA step."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import Dataset
from config import get_config
from dataset.data_loader.ConsecutiveClipDataset import ConsecutiveClipDataset, ExposureMatchedSampler, exposure_report, build_training_loader
from tools.bimamba.make_long_sequence_config import derive


class Clips(Dataset):
    data_format = 'NCDHW'
    def __init__(self):
        self.inputs = sorted('/cache/subject%d_input%d.npy' % (s, c)
                             for s in range(1, 31) for c in range(12 if s <= 24 else 9))
    def __len__(self):
        return len(self.inputs)
    def __getitem__(self, index):
        path = Path(self.inputs[index]).name
        subject = int(path.split('_')[0][7:])
        chunk = int(path.split('_input')[1][:-4])
        times = np.arange(160, dtype=np.float32) + chunk*160 + subject*10000
        return np.broadcast_to(times[None, :, None, None], (3,160,2,2)), times, str(subject), str(chunk)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cuda', action='store_true')
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--cache-dir', type=Path)
    ap.add_argument('--reference-config', type=Path)
    ap.add_argument('--report', type=Path)
    args = ap.parse_args()
    base = Clips()
    joined = ConsecutiveClipDataset(base)
    assert len(base) == 342 and len(joined) == 90
    for i in range(len(joined)):
        x, y, *_ = joined[i]
        assert x.shape == (3,640,2,2) and y.shape == (640,)
        np.testing.assert_array_equal(x[0,:,0,0], y)
        np.testing.assert_array_equal(np.diff(y), np.ones(639))
    sampler = ExposureMatchedSampler(joined, 30)
    assert sampler.plans == ExposureMatchedSampler(joined, 30).plans
    audit = exposure_report(joined, sampler, 4, 16)
    assert audit['reference_steps_per_epoch'] == audit['training_steps_per_epoch'] == 22
    assert audit['reference_frames_per_epoch'] == 54720 and audit['training_frames_per_epoch'] == 55040
    assert all(len(e['subjects']) == 30 and len(set(e['group_indices'])) == 86 for e in audit['epochs'])
    assert min(audit['cumulative_clip_uses'].values()) > 0
    for epoch in (0, 1, 29):
        sampler.set_epoch(epoch)
        assert list(sampler) == sampler.plans[epoch]
    # Shorter recordings retain approximately proportional cumulative exposure.
    for subject in set(joined.group_subjects):
        actual = sum(joined.group_subjects[g] == subject for plan in sampler.plans for g in plan)
        n = sum(Path(p).name.startswith(subject+'_') for p in base.inputs)
        assert abs(actual*4/(30*n)-1) < .05
    for mode in ('gap', 'duplicate', 'short', 'mixed_cache'):
        bad = Clips()
        if mode == 'gap': bad.inputs.remove('/cache/subject1_input5.npy')
        if mode == 'duplicate': bad.inputs.append(bad.inputs[0])
        if mode == 'short': bad.inputs = ['/cache/subject1_input%d.npy' % i for i in range(3)]
        if mode == 'mixed_cache': bad.inputs[0] = bad.inputs[0].replace('/cache/', '/other/')
        try:
            ConsecutiveClipDataset(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Should reject ' + mode)
    result = dict(synthetic_temporal_and_exposure_tests='passed',
                  reference_frames=54720, training_frames=55040, steps_per_epoch=22,
                  all_subjects_every_epoch=True)
    # Exercise the actual main.py loader helper, including its disabled default.
    root = Path(__file__).resolve().parents[1]
    cfg = get_config(SimpleNamespace(config_file=str(root/'configs/train_configs/UBFC_UBFC_BIMAMBA_STUDENT_NOKD_T640.yaml')))
    with tempfile.TemporaryDirectory() as td:
        temp = Path(td)
        cfg.defrost()
        cfg.MODEL.MODEL_DIR = str(temp/'models')
        cfg.NUM_WORKERS = 0
        for split in ('TRAIN','TEST'):
            path = temp/(split+'.csv'); path.write_text('input_files\n/example/placeholder.npy\n')
            cfg[split].DATA.FILE_LIST_PATH = str(path)
        cfg.freeze()
        loader = build_training_loader(base, cfg, None, torch.Generator().manual_seed(100))
        assert len(loader) == 22 and len(loader.dataset.inputs) == 342
        batch = next(iter(loader))
        assert tuple(batch[0].shape) == (4,3,640,2,2)
        assert tuple(batch[1].shape) == (4,640)
        assert (temp/'models/sequence_exposure.json').is_file()
        # Exercise the real epoch hook without expensive model construction/training.
        from neural_methods.trainer.BiMambaPhysTrainer import BiMambaPhysTrainer
        trainer = object.__new__(BiMambaPhysTrainer)
        trainer.max_epoch_num = 2
        trainer.model_dir = str(temp/'models')
        trainer.optimizer = SimpleNamespace(param_groups=[dict(lr=.0003)])
        trainer.config = SimpleNamespace(TRAIN=SimpleNamespace(PLOT_LOSSES_AND_LR=False))
        seen = []
        def run_epoch(data_loader, training):
            seen.append(list(data_loader.sampler))
            return dict(total=0.)
        trainer._run_epoch = run_epoch
        trainer.save_model = lambda epoch: None
        trainer.train(dict(train=loader, valid=None))
        assert seen == loader.sampler.plans[:2] and seen[0] != seen[1]
        cfg.defrost(); cfg.TRAIN.SEQUENCE_LENGTH = 0; cfg.freeze()
        short = build_training_loader(base, cfg, None, torch.Generator().manual_seed(100))
        assert short.dataset is base and next(iter(short))[0].shape[2] == 160
    result['main_loader_and_trainer_epoch_hook'] = 'passed, including disabled default and unchanged source membership'
    if args.cache_dir:
        from dataset.data_loader.BaseLoader import BaseLoader
        ds = object.__new__(BaseLoader)
        ds.inputs = sorted(str(p) for p in args.cache_dir.glob('subject12_input*.npy'))
        assert len(ds.inputs) >= 4
        ds.labels = [p.replace('_input', '_label') for p in ds.inputs]
        ds.data_format = 'NCDHW'
        real = ConsecutiveClipDataset(ds)
        x, y, *_ = real[0]
        for offset, original_index in enumerate(real.groups[0]):
            original = ds[original_index]
            np.testing.assert_array_equal(x[:,offset*160:(offset+1)*160], original[0])
            np.testing.assert_array_equal(y[offset*160:(offset+1)*160], original[1])
        result['real_cache_join'] = 'subject12 input0..3 byte-value equivalent after existing float32 conversion; no training'
    if args.reference_config:
        import yaml
        old = yaml.safe_load(args.reference_config.read_text())
        new = derive(old)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'long.yaml'; path.write_text(yaml.safe_dump(new))
            loaded = yaml.safe_load(get_config(SimpleNamespace(config_file=str(path))).dump())
        for split in ('TRAIN', 'TEST'):
            assert loaded[split]['DATA'] == old[split]['DATA'], split
        assert loaded['MODEL']['BIMAMBAPHYS'] == old['MODEL']['BIMAMBAPHYS']
        assert loaded['DISTILL'] == old['DISTILL'] and loaded['INFERENCE'] == old['INFERENCE']
        assert loaded['TRAIN']['EPOCHS'] == old['TRAIN']['EPOCHS']
        assert loaded['TRAIN']['LR'] == old['TRAIN']['LR']
        assert loaded['TEST']['METRICS'] == old['TEST']['METRICS']
        assert loaded['TEST']['USE_LAST_EPOCH'] == old['TEST']['USE_LAST_EPOCH']
        assert loaded['TRAIN']['BATCH_SIZE'] == 4 and loaded['TRAIN']['SEQUENCE_LENGTH'] == 640
        result['actual_reference_config_roundtrip'] = 'passed: cache, splits, architecture, losses and test settings identical'
    if args.cuda:
        from neural_methods.model.BiMambaPhys import BiMambaPhys
        from neural_methods.loss.BiMambaDistillationLoss import BiMambaDistillationLoss
        torch.set_num_threads(4); torch.manual_seed(100)
        assert torch.cuda.is_available()
        model = BiMambaPhys(dim=64, depth=6, dual_path=True).cuda().train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        criterion = BiMambaDistillationLoss(kd_weight=0)
        torch.cuda.reset_peak_memory_stats()
        x = torch.randn(args.batch_size,3,640,args.size,args.size,device='cuda')
        y = torch.randn(args.batch_size,640,device='cuda')
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            assert prediction.shape == y.shape
            loss, parts = criterion(prediction,y)
            assert torch.isfinite(loss) and parts['kd'].item() == 0
            loss.backward()
            grads = [p.grad for n,p in model.named_parameters() if 'mamba' in n.lower() and p.requires_grad]
            assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1, error_if_nonfinite=True)
            optimizer.step()
        result['cuda_smoke'] = dict(batch=args.batch_size, frames=640, size=args.size,
                                   two_optimizer_steps=True, peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
        model.eval()
        with torch.no_grad():
            assert model(x[:1,:,:160]).shape == (1,160)
        result['unchanged_test_length_forward'] = 'passed at T=160'
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
