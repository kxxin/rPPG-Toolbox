"""Adapted from PhysMambaTrainer: toolbox loops, metrics, and output format.

Two-stage version: MODEL.BIMAMBAPHYS.STAGE1_EPOCHS controls the DualPhysMamba
schedule (stage 1 = rPPG path only, stage 2 = joint). Validation and test always
use stage 2. STAGE1_EPOCHS=0 reproduces the previous single-stage trainer.
test() also reports the RhythmMamba/DualPhysMamba paper protocol
(evaluation/paper_protocol.py) when INFERENCE.PAPER_PROTOCOL is True.
"""
import json
import os
import re
import warnings

import torch
from tqdm import tqdm
from evaluation.metrics import calculate_metrics
from neural_methods.loss.RhythmMambaLoss import build_supervised_loss
from neural_methods.model.BiMambaPhys import BiMambaPhys
from neural_methods.trainer.BaseTrainer import BaseTrainer


class BiMambaPhysTrainer(BaseTrainer):
    def __init__(self, config, data_loader):
        super().__init__()
        self.config, self.device = config, torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir, self.model_file_name = config.MODEL.MODEL_DIR, config.TRAIN.MODEL_FILE_NAME
        self.best_epoch, self.min_valid_loss = 0, float("inf")
        self.learning_rates = []
        m, loss = config.MODEL.BIMAMBAPHYS, config.MODEL.BIMAMBAPHYS.LOSS
        # DualPhysMamba two-stage schedule: epochs [0, STAGE1_EPOCHS) train the rPPG
        # path only; remaining epochs train both paths. 0 = single-stage (old behaviour).
        self.stage1_epochs = int(getattr(m, "STAGE1_EPOCHS", 0))
        if self.stage1_epochs < 0:
            raise ValueError("STAGE1_EPOCHS must be >= 0.")
        self.noise_path = bool(getattr(m, "NOISE_PATH", True))
        self.arch = str(getattr(m, "ARCH", "bimamba")).lower()
        if self.arch in ("dualphysmamba", "bidualphysmamba") and not getattr(m, "DUAL_PATH", False):
            raise ValueError("ARCH %s is a dual-path model family; set DUAL_PATH=True." % self.arch)
        if self.arch in ("rhythmmamba", "rhythmformer") and (getattr(m, "DUAL_PATH", False) or self.stage1_epochs):
            raise ValueError("ARCH %s is single-path; set DUAL_PATH=False and STAGE1_EPOCHS=0." % self.arch)
        if self.arch == "dualphysmamba" and not self.noise_path:
            raise ValueError("NOISE_PATH=False is only implemented for ARCH bimamba and bidualphysmamba.")
        if not self.noise_path:
            if not getattr(m, "DUAL_PATH", False):
                raise ValueError("NOISE_PATH=False is a variant of the DUAL_PATH model; set DUAL_PATH=True.")
            if self.stage1_epochs:
                raise ValueError("NOISE_PATH=False has no stage 2; set STAGE1_EPOCHS=0.")
        if self.stage1_epochs > 0:
            if not getattr(m, "DUAL_PATH", False):
                raise ValueError("STAGE1_EPOCHS > 0 requires DUAL_PATH=True.")
            if config.TOOLBOX_MODE == "train_and_test" and self.stage1_epochs >= self.max_epoch_num:
                raise ValueError("STAGE1_EPOCHS must be smaller than TRAIN.EPOCHS so stage 2 is trained.")
        stages = ("TEST",)
        if config.TOOLBOX_MODE == "train_and_test":
            stages = ("TRAIN", "TEST") if config.TEST.USE_LAST_EPOCH else ("TRAIN", "VALID", "TEST")
            if not config.TEST.USE_LAST_EPOCH and not data_loader.get("valid"):
                raise ValueError("USE_LAST_EPOCH=False requires validation data")
        for stage in stages:
            data = config[stage].DATA
            if data.PREPROCESS.DATA_TYPE != ["Standardized"] or data.PREPROCESS.LABEL_TYPE != "Standardized":
                raise ValueError("This baseline requires Standardized RGB and labels for every split.")
            if data.DATA_FORMAT != "NCDHW":
                raise ValueError("BiMambaPhys requires DATA_FORMAT: NCDHW.")
        self.model = BiMambaPhys(dim=m.DIM, depth=m.DEPTH, d_state=m.D_STATE,
                                d_conv=m.D_CONV, expand=m.EXPAND, dropout=m.DROPOUT,
                                dual_path=getattr(m, 'DUAL_PATH', False),
                                mlp_ratio=getattr(m, 'MLP_RATIO', 2),
                                drop_path_rate=getattr(m, 'DROP_PATH_RATE', 0.1),
                                cross_attn_heads=getattr(m, 'CROSS_ATTN_HEADS', 4),
                                noise_path=self.noise_path, arch=self.arch).to(self.device)
        if config.NUM_OF_GPU_TRAIN > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        fs = config.TRAIN.DATA.FS if config.TOOLBOX_MODE == "train_and_test" else config.TEST.DATA.FS
        # LOSS.TYPE: 'bimamba' (Pearson + L1 + spectral JS, default) or 'rhythmmamba' (0.2 Pearson + 1.0 CE).
        self.criterion = build_supervised_loss(config, fs).to(self.device)
        if config.TOOLBOX_MODE == "train_and_test":
            if not data_loader.get("train") or len(data_loader["train"]) == 0:
                raise ValueError("Training loader is empty.")
            self._audit_splits(data_loader)
            self._check_manifests(data_loader)
            # Respect Mamba's no-weight-decay flags for A_log and the D skip term.
            decay, no_decay = [], []
            for parameter in self.model.parameters():
                (no_decay if getattr(parameter, "_no_weight_decay", False) else decay).append(parameter)
            self.optimizer = torch.optim.AdamW([
                {"params": decay, "weight_decay": m.WEIGHT_DECAY},
                {"params": no_decay, "weight_decay": 0.0},
            ], lr=config.TRAIN.LR)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR, epochs=self.max_epoch_num,
                steps_per_epoch=len(data_loader["train"]))
            os.makedirs(self.model_dir, exist_ok=True)
            with open(os.path.join(self.model_dir, "resolved_config.yaml"), "w") as handle:
                handle.write(config.dump())
            if getattr(self, "split_record", None):
                with open(os.path.join(self.model_dir, "split_record.json"), "w") as handle:
                    json.dump(self.split_record, handle, indent=2)
        elif config.TOOLBOX_MODE == "only_test":
            self._check_manifests(data_loader)
        else:
            raise ValueError("BiMambaPhys supports train_and_test and only_test.")
        print("BiMambaPhys parameters (ARCH %s):" % self.arch, sum(p.numel() for p in self.model.parameters()))

    def _audit_splits(self, loaders):
        subjects = {}
        for name, loader in loaders.items():
            if loader is not None and hasattr(loader.dataset, "inputs"):
                subjects[name] = {os.path.basename(p).rsplit("_input", 1)[0] for p in loader.dataset.inputs}
        for name in ("valid", "test"):
            if subjects.get("train", set()) & subjects.get(name, set()):
                raise ValueError("Subject leakage between train and " + name)
        if subjects.get("valid", set()) & subjects.get("test", set()):
            warnings.warn("Validation and test subjects overlap as configured. Test metrics are development results, not independent held-out estimates.")

    @staticmethod
    def _subject_id(dataset, stem):
        if dataset == "PURE":
            return int(stem) // 100
        match = re.fullmatch(r"subject(\d+)", stem)
        if not match:
            raise ValueError("Cannot parse subject from cache name %r" % stem)
        return int(match.group(1))

    def _check_manifests(self, loaders):
        """Fail fast if the loaded subjects differ from the pinned split manifests.

        The toolbox silently regenerates a missing FILE_LIST_PATH CSV from the unsorted
        glob order; this check turns that into an error instead of a wrong test set.
        """
        manifests = list(getattr(self.config.MODEL.BIMAMBAPHYS, "SPLIT_MANIFESTS", []) or [])
        if not manifests:
            return
        record = {}
        for path in manifests:
            with open(path) as handle:
                manifest = json.load(handle)
            for split, info in manifest["splits"].items():
                loader = loaders.get(split)
                section = self.config[split.upper()].DATA if split.upper() in self.config else None
                if loader is None or section is None or section.DATASET != manifest["dataset"]:
                    continue
                loaded = sorted({self._subject_id(manifest["dataset"], os.path.basename(p).rsplit("_input", 1)[0])
                                 for p in loader.dataset.inputs})
                if loaded != sorted(info["subjects"]):
                    raise ValueError("%s split differs from %s: loaded %s, manifest %s. Check FILE_LIST_PATH."
                                     % (split, path, loaded, sorted(info["subjects"])))
                if len(loader.dataset.inputs) != info["clips"]:
                    raise ValueError("%s clip count %d != manifest %d (%s)" %
                                     (split, len(loader.dataset.inputs), info["clips"], path))
                record[split] = dict(manifest=path, dataset=manifest["dataset"], subjects=loaded,
                                     clips=info["clips"], csv_sha256=info["sha256"])
        if not record:
            raise ValueError("SPLIT_MANIFESTS set but no loaded split matched any manifest.")
        print("Pinned split verified:", {k: v["subjects"] for k, v in record.items()}, flush=True)
        self.split_record = record

    def _stage(self, epoch):
        return 1 if epoch < self.stage1_epochs else 2

    def _run_epoch(self, loader, training, stage=2):
        if loader is None or len(loader) == 0:
            raise ValueError("Missing or empty data loader.")
        self.model.train(training)
        totals = {}
        count = 0
        with torch.set_grad_enabled(training):
            progress = tqdm(loader, ncols=100, desc=("Train S%d" % stage) if training else "Valid")
            for batch in progress:
                data, target = batch[0].float().to(self.device), batch[1].float().to(self.device)
                if training:
                    # set_to_none: in stage 1 the noise path keeps grad=None, so AdamW
                    # skips it entirely (no update, no weight decay).
                    self.optimizer.zero_grad(set_to_none=True)
                prediction = self.model(data, stage=stage)
                value, parts = self.criterion(prediction, target, return_components=True)
                if not torch.isfinite(value):
                    raise FloatingPointError("Non-finite waveform loss.")
                if training:
                    value.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.MODEL.BIMAMBAPHYS.GRAD_CLIP,
                                                   error_if_nonfinite=True)
                    self.learning_rates.append(self.optimizer.param_groups[0]["lr"])
                    self.optimizer.step()
                    self.scheduler.step()
                size = data.shape[0]
                count += size
                for key, component in dict(total=value, **parts).items():
                    totals[key] = totals.get(key, 0.0) + component.detach().item() * size
                progress.set_postfix(loss=value.item())
        return {key: value / count for key, value in totals.items()}

    def train(self, data_loader):
        history, train_losses, valid_losses = [], [], []
        for epoch in range(self.max_epoch_num):
            if hasattr(data_loader['train'].sampler, 'set_epoch'):
                data_loader['train'].sampler.set_epoch(epoch)
            stage = self._stage(epoch)
            print("\n==== Training Epoch: %d / %d (stage %d) ====" % (epoch + 1, self.max_epoch_num, stage), flush=True)
            lr = self.optimizer.param_groups[0]["lr"]
            train = self._run_epoch(data_loader["train"], True, stage=stage)
            # Validation always scores the full stage-2 model, as in the reproduction.
            valid = self._run_epoch(data_loader["valid"], False, stage=2) if data_loader.get("valid") is not None else None
            self.save_model(epoch)
            # Only stage-2 epochs are eligible for best-epoch selection.
            if valid is not None and stage == 2 and valid["total"] < self.min_valid_loss:
                self.min_valid_loss, self.best_epoch = valid["total"], epoch
            history.append({"epoch": epoch, "stage": stage, "lr": lr, "train": train, "valid": valid})
            with open(os.path.join(self.model_dir, "history.json"), "w") as handle:
                json.dump(history, handle, indent=2)
            train_losses.append(train["total"])
            if valid is not None:
                valid_losses.append(valid["total"])
            print(json.dumps(history[-1]), flush=True)
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(train_losses, valid_losses, self.learning_rates, self.config)

    def valid(self, data_loader):
        return self._run_epoch(data_loader["valid"], False)["total"]

    def _checkpoint(self, epoch):
        return os.path.join(self.model_dir, self.model_file_name + "_Epoch%d.pth" % epoch)

    def save_model(self, index):
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        torch.save(model.state_dict(), self._checkpoint(index))

    def test(self, data_loader):
        if data_loader.get("test") is None:
            raise ValueError("Missing test data.")
        if self.config.TOOLBOX_MODE == "only_test":
            checkpoint = self.config.INFERENCE.MODEL_PATH
        else:
            epoch = self.max_epoch_num - 1 if self.config.TEST.USE_LAST_EPOCH else self.best_epoch
            checkpoint = self._checkpoint(epoch)
        model = self.model.module if isinstance(self.model, torch.nn.DataParallel) else self.model
        model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        print("Testing checkpoint:", checkpoint, flush=True)
        os.makedirs(self.config.TEST.OUTPUT_SAVE_DIR, exist_ok=True)
        with open(os.path.join(self.config.TEST.OUTPUT_SAVE_DIR, "checkpoint_selection.json"), "w") as handle:
            json.dump(dict(checkpoint=checkpoint, best_epoch_zero_based=self.best_epoch,
                           selection="configured checkpoint" if self.config.TOOLBOX_MODE == "only_test" else
                           ("last epoch" if self.config.TEST.USE_LAST_EPOCH else "minimum validation total loss"),
                           independent_test=False), handle, indent=2)
        self.model.eval()
        predictions, labels = {}, {}
        with torch.no_grad():
            for batch in tqdm(data_loader["test"], desc="Test", ncols=100):
                output = self.model(batch[0].float().to(self.device)).cpu()
                for i in range(output.shape[0]):
                    subject, chunk = str(batch[2][i]), int(batch[3][i])
                    predictions.setdefault(subject, {})[chunk] = output[i]
                    labels.setdefault(subject, {})[chunk] = batch[1][i].float().cpu()
        # Save before evaluation, so waveforms survive a metric/plotting failure.
        self.save_test_outputs(predictions, labels, self.config)
        from tools.bimamba.report_waveforms import report_waveforms
        report_waveforms(predictions, labels, self.config.TEST.DATA.FS,
                         self.config.TEST.OUTPUT_SAVE_DIR, checkpoint)
        if getattr(self.config.INFERENCE, "PAPER_PROTOCOL", True):
            # Before the toolbox metrics so a plotting failure there cannot lose it.
            from evaluation.paper_protocol import report as paper_report
            paper_report(predictions, labels, self.config, self.config.TEST.OUTPUT_SAVE_DIR)
        print("\n==== Toolbox protocol (periodogram whole video, 0.6-3.3 Hz, SNR 10log10) ====")
        calculate_metrics(predictions, labels, self.config)
