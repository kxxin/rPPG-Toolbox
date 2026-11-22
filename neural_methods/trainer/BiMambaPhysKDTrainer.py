"""Frozen teacher(s) -> independently initialized 6-block student.

Single teacher: the 12+12 dual-path BiMambaPhys teacher (DISTILL.TEACHER_CHECKPOINT/CONFIG), output and/or
feature KD. Multi-teacher (DISTILL.TEACHER_RUNS non-empty): any trained ARCH (bimamba, dualphysmamba,
bidualphysmamba, rhythmmamba); output KD only, towards the mean of the teachers' z-scored waveforms.

Inherits checkpointing, train loop, waveform reporting and test evaluation from
BiMambaPhysTrainer. Only _run_epoch performs distillation; only_test never loads
or requires a teacher. Model checkpoints contain student state exclusively.
"""
import csv
import hashlib
import json
from pathlib import Path
import torch
import yaml
from tqdm import tqdm
from neural_methods.model.BiMambaPhys import BiMambaPhys
from neural_methods.loss.BiMambaDistillationLoss import BiMambaDistillationLoss
from neural_methods.loss.FeatureDistillationLoss import build_feature_loss, block_mapping
from neural_methods.trainer.BiMambaPhysTrainer import BiMambaPhysTrainer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def clip_names(paths):
    names = [str(p).replace('\\', '/').rsplit('/', 1)[-1] for p in paths]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate subject/clip names in split')
    return set(names)


def _zscore(x):
    return (x - x.mean(-1, keepdim=True)) / (x.std(-1, keepdim=True) + 1e-8)


def build_from_config(tm, dual_path=True):
    """BiMambaPhys of any ARCH from a resolved_config MODEL.BIMAMBAPHYS dict."""
    return BiMambaPhys(dim=tm['DIM'], depth=tm['DEPTH'], d_state=tm.get('D_STATE', 16),
                       d_conv=tm.get('D_CONV', 4), expand=tm.get('EXPAND', 2), dropout=tm.get('DROPOUT', 0.1),
                       dual_path=dual_path, mlp_ratio=tm.get('MLP_RATIO', 2),
                       drop_path_rate=tm.get('DROP_PATH_RATE', .1), cross_attn_heads=tm.get('CROSS_ATTN_HEADS', 4),
                       noise_path=tm.get('NOISE_PATH', True), arch=str(tm.get('ARCH', 'bimamba')).lower())


class TeacherEnsemble(torch.nn.Module):
    """Frozen teachers; forward = mean of their z-scored stage-2 waveforms (re-z-scored)."""

    def __init__(self, teachers):
        super().__init__()
        self.members = torch.nn.ModuleList(teachers)

    def forward(self, video, stage=2):
        outs = [_zscore(t(video, stage=2)) for t in self.members]
        return _zscore(torch.stack(outs).mean(0))


def resolve_teacher_run(run):
    """(checkpoint, resolved_config) a teacher run was TESTED with."""
    run = Path(run).expanduser()
    configs = sorted(run.glob('*/PreTrainedModels/resolved_config.yaml'))
    if len(configs) != 1:
        raise FileNotFoundError('Expected exactly one */PreTrainedModels/resolved_config.yaml in teacher run %s, found %d'
                                % (run, len(configs)))
    with configs[0].open() as handle:
        tcfg = yaml.safe_load(handle)
    folder = configs[0].parent
    if tcfg['TEST'].get('USE_LAST_EPOCH', True):
        checkpoint = folder / ('%s_Epoch%d.pth' % (tcfg['TRAIN']['MODEL_FILE_NAME'], int(tcfg['TRAIN']['EPOCHS']) - 1))
    else:
        selection = run / tcfg['TEST']['DATA']['EXP_DATA_NAME'] / 'saved_test_outputs' / 'checkpoint_selection.json'
        if not selection.is_file():
            raise FileNotFoundError('Teacher %s has no checkpoint_selection.json (not tested yet?): %s' % (run, selection))
        # the file stores an absolute path; use the same file name inside this run (robust to moved folders)
        checkpoint = folder / Path(json.loads(selection.read_text())['checkpoint']).name
    if not checkpoint.is_file():
        raise FileNotFoundError('Teacher checkpoint missing: %s' % checkpoint)
    return checkpoint, configs[0]


class BiMambaPhysKDTrainer(BiMambaPhysTrainer):
    def __init__(self, config, data_loader):
        m, loss, kd = config.MODEL.BIMAMBAPHYS, config.MODEL.BIMAMBAPHYS.LOSS, config.DISTILL
        if not m.DUAL_PATH or m.DEPTH != 6:
            raise ValueError('This experiment requires DUAL_PATH=True and DEPTH=6 per student path')
        super().__init__(config, data_loader)
        fs = config.TRAIN.DATA.FS if config.TOOLBOX_MODE == 'train_and_test' else config.TEST.DATA.FS
        self.distillation = BiMambaDistillationLoss(fs, loss.PEARSON_WEIGHT, loss.WAVE_WEIGHT,
                                                   loss.FREQUENCY_WEIGHT, loss.MIN_HZ, loss.MAX_HZ,
                                                   kd_weight=kd.WEIGHT,
                                                   supervised=self.criterion)  # LOSS.TYPE-selected GT loss
        # Feature KD (DISTILL.FEATURE.*_WEIGHT); independent of the output-KD weight DISTILL.WEIGHT.
        self.feature_kd = build_feature_loss(config)
        self.teacher = None
        if config.TOOLBOX_MODE == 'train_and_test':
            metadata = dict(student_depth_per_path=6, student_noise_path=bool(getattr(m, 'NOISE_PATH', True)), student_initialization='random, no teacher weight copying',
                            student_parameters=sum(p.numel() for p in self.model.parameters()),
                            kd_weight=kd.WEIGHT, distillation='normalized waveform Pearson + L1 + spectral JS',
                            supervised_loss=getattr(loss, 'TYPE', 'bimamba'),
                            teacher_loaded=False, teacher_used_for_validation_or_test=False)
            metadata['feature_kd'] = dict(self.feature_kd.weights, active=self.feature_kd.active)
            teacher_runs = list(getattr(kd, 'TEACHER_RUNS', []) or [])
            if teacher_runs:
                if self.feature_kd.active:
                    raise ValueError('Multi-teacher KD (DISTILL.TEACHER_RUNS) supports output KD only; set DISTILL.FEATURE.* to 0')
                if kd.WEIGHT <= 0:
                    raise ValueError('DISTILL.TEACHER_RUNS is set but DISTILL.WEIGHT is 0')
                metadata.update(self._load_teachers(teacher_runs, data_loader))
                metadata['distillation'] = 'waveform KD (Pearson + L1 + spectral JS) towards mean of z-scored teacher waveforms'
            elif kd.WEIGHT > 0 or self.feature_kd.active:
                metadata.update(self._load_teacher(data_loader))
                if self.feature_kd.active:
                    metadata['feature_kd']['student_to_teacher_block'] = block_mapping(
                        len(self.model.dual.blocks_rppg), len(self.teacher.dual.blocks_rppg))
            (Path(self.model_dir)/'distillation_setup.json').write_text(json.dumps(metadata, indent=2))

    def _load_teachers(self, runs, loaders):
        members, info = [], []
        for run in runs:
            checkpoint, config_path = resolve_teacher_run(run)
            with config_path.open() as handle:
                teacher_cfg = yaml.safe_load(handle)
            audit = self._audit_teacher_splits(teacher_cfg, loaders)
            with torch.random.fork_rng(devices=[]):  # must not perturb the student's RNG
                teacher = build_from_config(teacher_cfg['MODEL']['BIMAMBAPHYS'],
                                            dual_path=teacher_cfg['MODEL']['BIMAMBAPHYS'].get('DUAL_PATH', False))
            teacher.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            teacher.requires_grad_(False)
            members.append(teacher.eval())
            info.append(dict(run=str(run), arch=str(teacher_cfg['MODEL']['BIMAMBAPHYS'].get('ARCH', 'bimamba')),
                             checkpoint=str(checkpoint.resolve()), checkpoint_sha256=sha256(checkpoint),
                             config=str(config_path.resolve()), parameters=sum(p.numel() for p in teacher.parameters()),
                             matched_splits=audit))
            print('Frozen teacher loaded:', checkpoint, flush=True)
        self.teacher = TeacherEnsemble(members).to(self.device).eval()
        optimizer_ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        if any(id(p) in optimizer_ids for p in self.teacher.parameters()):
            raise RuntimeError('Teacher parameters must never be in the student optimizer')
        return dict(teacher_loaded=True, teachers=info, n_teachers=len(members),
                    ensemble='mean of per-clip z-scored stage-2 waveforms, re-z-scored')

    def _audit_teacher_splits(self, teacher_cfg, loaders):
        cfg = self.config
        split_audit = {}
        for split in ('TRAIN', 'TEST'):
            td, sd = teacher_cfg[split]['DATA'], cfg[split].DATA
            if td['FS'] != sd.FS or td['DATA_FORMAT'] != sd.DATA_FORMAT:
                raise ValueError('Teacher/student sampling rate or data layout differs: ' + split)
            pre = td['PREPROCESS']
            for key in ('DATA_TYPE', 'LABEL_TYPE', 'CHUNK_LENGTH', 'DO_CHUNK'):
                if pre[key] != sd.PREPROCESS[key]:
                    raise ValueError('Teacher/student preprocessing mismatch: %s.%s' % (split, key))
            if pre['RESIZE']['H'] != sd.PREPROCESS.RESIZE.H or pre['RESIZE']['W'] != sd.PREPROCESS.RESIZE.W:
                raise ValueError('Teacher/student image size differs: ' + split)
            file_list = Path(td['FILE_LIST_PATH']).expanduser()
            if not file_list.is_file():
                raise FileNotFoundError('Teacher split CSV is unavailable: %s. If relocated, update its path in a copy of the teacher resolved config.' % file_list)
            with file_list.open(newline='') as handle:
                teacher_names = clip_names(row['input_files'] for row in csv.DictReader(handle))
            loader = loaders.get(split.lower())
            if loader is None or not hasattr(loader.dataset, 'inputs'):
                raise ValueError('Need dataset.inputs to verify the actual loaded ' + split + ' split')
            student_names = clip_names(loader.dataset.inputs)
            if not teacher_names or teacher_names != student_names:
                raise ValueError('Teacher/student %s clip membership differs (missing=%s, extra=%s)' %
                                 (split, sorted(teacher_names-student_names)[:8], sorted(student_names-teacher_names)[:8]))
            split_audit[split.lower()] = dict(clips=len(student_names),
                                             subjects=sorted({name.rsplit('_input', 1)[0] for name in student_names}),
                                             csv_sha256=sha256(file_list))
        if set(split_audit['train']['subjects']) & set(split_audit['test']['subjects']):
            raise ValueError('Teacher train and student test subjects overlap')
        return split_audit

    def _load_teacher(self, loaders):
        cfg = self.config
        checkpoint, config_path = Path(cfg.DISTILL.TEACHER_CHECKPOINT).expanduser(), Path(cfg.DISTILL.TEACHER_CONFIG).expanduser()
        for path in (checkpoint, config_path):
            if not path.is_file():
                raise FileNotFoundError('Frozen teacher file missing: %s. Set DISTILL.TEACHER_CHECKPOINT and TEACHER_CONFIG in the KD YAML.' % path)
        with config_path.open() as handle:
            teacher_cfg = yaml.safe_load(handle)
        tm = teacher_cfg['MODEL']['BIMAMBAPHYS']
        if not tm.get('DUAL_PATH', False) or tm['DEPTH'] != 12:
            raise ValueError('Teacher resolved_config.yaml must describe the trained 12+12 dual model')
        split_audit = self._audit_teacher_splits(teacher_cfg, loaders)
        # Teacher construction must not perturb the student's training RNG compared with the control.
        with torch.random.fork_rng(devices=[]):
            teacher = BiMambaPhys(dim=tm['DIM'], depth=tm['DEPTH'], d_state=tm['D_STATE'],
                                  d_conv=tm['D_CONV'], expand=tm['EXPAND'], dropout=tm['DROPOUT'],
                                  dual_path=True, mlp_ratio=tm.get('MLP_RATIO', 2),
                                  drop_path_rate=tm.get('DROP_PATH_RATE', .1),
                                  cross_attn_heads=tm.get('CROSS_ATTN_HEADS', 4),
                                  noise_path=tm.get('NOISE_PATH', True))
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        # Strict loading fails on a baseline checkpoint, wrong depth, or incompatible model,
        # including teachers trained before the spatial attention_conv was added.
        teacher.load_state_dict(state, strict=True)
        teacher.requires_grad_(False)
        self.teacher = teacher.to(self.device).eval()
        optimizer_ids = {id(p) for group in self.optimizer.param_groups for p in group['params']}
        if any(id(p) in optimizer_ids for p in self.teacher.parameters()):
            raise RuntimeError('Teacher parameters must never be in the student optimizer')
        print('Frozen teacher loaded:', checkpoint, flush=True)
        return dict(teacher_loaded=True, teacher_depth_per_path=12,
                    teacher_parameters=sum(p.numel() for p in self.teacher.parameters()),
                    teacher_checkpoint=str(checkpoint.resolve()), teacher_checkpoint_sha256=sha256(checkpoint),
                    teacher_config=str(config_path.resolve()), teacher_config_sha256=sha256(config_path),
                    matched_splits=split_audit)

    def _run_epoch(self, loader, training, stage=2):
        # Student follows the STAGE1_EPOCHS schedule; the frozen teacher is always
        # its full stage-2 model.
        if loader is None or len(loader) == 0:
            raise ValueError('Missing or empty data loader')
        self.model.train(training)
        if self.teacher is not None:
            # Keep Dropout/DropPath off and ALL BatchNorm buffers frozen.
            self.teacher.eval()
        totals, count = {}, 0
        with torch.set_grad_enabled(training):
            progress = tqdm(loader, ncols=100, desc=('Student train S%d' % stage) if training else 'Student valid')
            for batch in progress:
                video, target = batch[0].float().to(self.device), batch[1].float().to(self.device)
                teacher_prediction, teacher_aux = None, None
                use_features = training and self.teacher is not None and self.feature_kd.active
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.teacher is not None:
                        with torch.no_grad():
                            if use_features:
                                teacher_prediction, teacher_aux = self.teacher.dual(video, stage=2, return_aux=True)
                            else:
                                teacher_prediction = self.teacher(video, stage=2)
                if use_features:
                    student_prediction, student_aux = self.model.dual(video, stage=stage, return_aux=True)
                else:
                    student_prediction = self.model(video, stage=stage)
                # Output KD only if DISTILL.WEIGHT > 0 (the teacher may be loaded for feature KD alone).
                value, parts = self.distillation(student_prediction, target,
                                                 teacher_prediction if self.distillation.kd_weight > 0 else None)
                if use_features:
                    feature_total, feature_parts = self.feature_kd(student_aux, teacher_aux)
                    value = value + feature_total
                    parts = dict(parts, feat=feature_total, **{'feat_' + k: v for k, v in feature_parts.items()})
                if not torch.isfinite(value):
                    raise FloatingPointError('Non-finite distillation loss')
                if training:
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MODEL.BIMAMBAPHYS.GRAD_CLIP,
                                                   error_if_nonfinite=True)
                    self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
                    self.optimizer.step(); self.scheduler.step()
                size = video.shape[0]; count += size
                for key, component in dict(total=value, **parts).items():
                    totals[key] = totals.get(key, 0.) + component.detach().item() * size
                progress.set_postfix(loss=value.item(), gt=parts['supervised'].item(), kd=parts['kd'].item(),
                                     feat=parts['feat'].item() if 'feat' in parts else 0.0)
        return {key: value/count for key, value in totals.items()}
