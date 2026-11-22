"""CUDA integration smoke test with a temporary RANDOM teacher, not a training result."""
import argparse
import csv
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from types import SimpleNamespace
import torch
import yaml
from torch.utils.data import Dataset, DataLoader
from config import get_config
from neural_methods.model.BiMambaPhys import BiMambaPhys
from neural_methods.loss.BiMambaDistillationLoss import BiMambaDistillationLoss
from neural_methods.trainer.BiMambaPhysKDTrainer import BiMambaPhysKDTrainer


class SyntheticClips(Dataset):
    def __init__(self, subject, batch_size, size):
        self.inputs = ['/%s_input%d.npy' % (subject, i) for i in range(batch_size * 2)]
        self.video = torch.randn(len(self.inputs), 3, 160, size, size)
        self.target = torch.randn(len(self.inputs), 160)
    def __len__(self): return len(self.inputs)
    def __getitem__(self, index): return self.video[index], self.target[index]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--size', type=int, default=32)
    ap.add_argument('--batch-size', type=int, default=2)
    ap.add_argument('--report', type=Path)
    args = ap.parse_args()
    torch.manual_seed(100); torch.set_num_threads(4)
    if not torch.cuda.is_available(): raise RuntimeError('Actual Mamba CUDA kernels required')
    root = Path(__file__).resolve().parents[1]
    # Direct loss test: teacher tensor must be detached even if caller forgets no_grad.
    s, t, y = torch.randn(2,160,requires_grad=True), torch.randn(2,160,requires_grad=True), torch.randn(2,160)
    criterion = BiMambaDistillationLoss()
    value, parts = criterion(s,y,t)
    torch.testing.assert_close(value, parts['supervised'] + .5*parts['kd'])
    value.backward(); assert t.grad is None and s.grad.abs().sum() > 0
    torch.testing.assert_close(criterion(s,y,s)[1]['kd'], torch.zeros(()), atol=1e-6, rtol=0)
    train = SyntheticClips('subject1', args.batch_size, args.size)
    test = SyntheticClips('subject12', args.batch_size, args.size)
    loaders = dict(train=DataLoader(train,batch_size=args.batch_size), valid=None,
                   test=DataLoader(test,batch_size=args.batch_size))
    cfg = get_config(SimpleNamespace(config_file=str(root/'configs/train_configs/UBFC_UBFC_BIMAMBA_KD.yaml')))
    with tempfile.TemporaryDirectory(prefix='bimamba_kd_smoke_') as td:
        temp = Path(td)
        teacher_cfg = yaml.safe_load(cfg.dump())
        teacher_cfg['MODEL']['NAME'] = 'BiMambaPhys'
        teacher_cfg['MODEL']['BIMAMBAPHYS']['DEPTH'] = 12
        for split, ds in (('TRAIN',train), ('TEST',test)):
            file_list = temp/(split+'.csv')
            with file_list.open('w',newline='') as f:
                w=csv.writer(f); w.writerow(['input_files']); w.writerows([[p] for p in ds.inputs])
            teacher_cfg[split]['DATA']['FILE_LIST_PATH'] = str(file_list)
        teacher_config = temp/'teacher_resolved.yaml'; teacher_config.write_text(yaml.safe_dump(teacher_cfg))
        teacher = BiMambaPhys(depth=12,dual_path=True)
        teacher_count = sum(p.numel() for p in teacher.parameters())
        torch.save(teacher.state_dict(),temp/'teacher.pth'); del teacher
        cfg.defrost(); cfg.MODEL.MODEL_DIR=str(temp/'student');cfg.DISTILL.TEACHER_CHECKPOINT=str(temp/'teacher.pth');cfg.DISTILL.TEACHER_CONFIG=str(teacher_config);cfg.freeze()
        runner = BiMambaPhysKDTrainer(cfg,loaders)
        assert len(runner.teacher.dual.blocks_rppg)==12 and len(runner.model.dual.blocks_rppg)==6
        assert len(runner.model.dual.blocks_noise)==6
        teacher_before = {k:v.detach().cpu().clone() for k,v in runner.teacher.state_dict().items()}
        first = next(runner.model.parameters()); student_before=first.detach().clone()
        calls=[];hook=runner.teacher.register_forward_hook(lambda *a: calls.append(1))
        train_result = runner._run_epoch(loaders['train'],True)
        assert len(calls)==len(loaders['train'])
        assert not torch.equal(first,student_before)
        assert not runner.teacher.training and all(not p.requires_grad and p.grad is None for p in runner.teacher.parameters())
        assert all(not module.training for module in runner.teacher.modules())
        for k,v in runner.teacher.state_dict().items():
            assert torch.equal(v.cpu(),teacher_before[k]), 'Teacher changed: '+k
        for path in (runner.model.dual.blocks_rppg,runner.model.dual.blocks_noise):
            for block in path:
                for branch in (block.attn.forward_mamba,block.attn.backward_mamba):
                    assert branch.in_proj.weight.grad is not None and branch.in_proj.weight.grad.abs().sum()>0
        validation = runner._run_epoch(loaders['test'],False)
        assert len(calls)==len(loaders['train']) and validation['kd']==0.
        hook.remove()
        runner.save_model(0)
        state=torch.load(runner._checkpoint(0),map_location='cpu',weights_only=True)
        assert set(state)==set(runner.model.state_dict()) and not any('teacher' in k for k in state)
        assert 'dual.blocks_rppg.6.attn.norm.weight' not in state
        count=sum(p.numel() for p in runner.model.parameters())
        with torch.no_grad():
            probe=train.video[:1].cuda(); expected=runner.model.eval()(probe).cpu()
        del runner, state; torch.cuda.empty_cache()
        # only_test must not touch teacher files, including after they become unavailable.
        cfg.defrost();cfg.TOOLBOX_MODE='only_test';cfg.DISTILL.TEACHER_CHECKPOINT='missing';cfg.DISTILL.TEACHER_CONFIG='missing';cfg.freeze()
        inference=BiMambaPhysKDTrainer(cfg,{'test':loaders['test']})
        assert inference.teacher is None
        inference.model.load_state_dict(torch.load(temp/'student/UBFC_UBFC_BiMambaPhys_StudentKD_Epoch0.pth',map_location='cpu',weights_only=True),strict=True)
        with torch.no_grad():torch.testing.assert_close(inference.model.eval()(probe).cpu(),expected)
        del inference;torch.cuda.empty_cache()
        # Missing teacher must fail for positive-weight training, never silently train from scratch.
        cfg.defrost();cfg.TOOLBOX_MODE='train_and_test';cfg.freeze()
        try: BiMambaPhysKDTrainer(cfg,loaders)
        except FileNotFoundError: pass
        else: raise AssertionError('Missing teacher did not fail')
        # The matched no-KD control deliberately skips loading the absent teacher.
        cfg.defrost();cfg.DISTILL.WEIGHT=0.;cfg.freeze()
        control=BiMambaPhysKDTrainer(cfg,loaders);assert control.teacher is None
        control_result=control._run_epoch(loaders['test'],False)
        assert control_result['kd']==0. and control_result['total']==control_result['supervised']
        report=dict(teacher_parameters=teacher_count, student_parameters=count,
                    reduction_percent=100*(1-count/teacher_count),teacher_state_unchanged=True,
                    teacher_gradients_absent=True, student_updated=True, optimizer_contains_only_student=True,
                    teacher_used_only_on_train=True,student_only_checkpoint_and_reload=True,
                    missing_teacher_rejected=True,no_kd_control_passed=True,
                    input_shape=[args.batch_size,3,160,args.size,args.size],
                    peak_cuda_allocated_mb=torch.cuda.max_memory_allocated()/2**20,
                    synthetic_train_loss=train_result, trained_teacher_validation=False)
    print(json.dumps(report,indent=2))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
