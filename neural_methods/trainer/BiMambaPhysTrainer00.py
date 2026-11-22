"""Adapted from PhysMambaTrainer: toolbox loops, metrics, and output format."""
import json
import os
import warnings

import torch
from tqdm import tqdm
from evaluation.metrics import calculate_metrics
from neural_methods.loss.BiMambaPhysLoss import BiMambaPhysLoss
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
        stages = ("TRAIN", "VALID", "TEST") if config.TOOLBOX_MODE == "train_and_test" else ("TEST",)
        for stage in stages:
            data = config[stage].DATA
            if data.PREPROCESS.DATA_TYPE != ["Standardized"] or data.PREPROCESS.LABEL_TYPE != "Standardized":
                raise ValueError("This baseline requires Standardized RGB and labels for every split.")
            if data.DATA_FORMAT != "NCDHW":
                raise ValueError("BiMambaPhys requires DATA_FORMAT: NCDHW.")
        self.model = BiMambaPhys(dim=m.DIM, depth=m.DEPTH, d_state=m.D_STATE,
                                d_conv=m.D_CONV, expand=m.EXPAND, dropout=m.DROPOUT).to(self.device)
        if config.NUM_OF_GPU_TRAIN > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(config.NUM_OF_GPU_TRAIN)))
        fs = config.TRAIN.DATA.FS if config.TOOLBOX_MODE == "train_and_test" else config.TEST.DATA.FS
        self.criterion = BiMambaPhysLoss(fs, loss.PEARSON_WEIGHT, loss.WAVE_WEIGHT,
                                       loss.FREQUENCY_WEIGHT, loss.MIN_HZ, loss.MAX_HZ)
        if config.TOOLBOX_MODE == "train_and_test":
            if not data_loader.get("train") or len(data_loader["train"]) == 0:
                raise ValueError("Training loader is empty.")
            self._audit_splits(data_loader)
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
        elif config.TOOLBOX_MODE != "only_test":
            raise ValueError("BiMambaPhys supports train_and_test and only_test.")
        print("BiMambaPhys parameters:", sum(p.numel() for p in self.model.parameters()))

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

    def _run_epoch(self, loader, training):
        if loader is None or len(loader) == 0:
            raise ValueError("Missing or empty data loader.")
        self.model.train(training)
        totals = dict(total=0.0, pearson=0.0, wave=0.0, frequency=0.0)
        count = 0
        with torch.set_grad_enabled(training):
            progress = tqdm(loader, ncols=100, desc="Train" if training else "Valid")
            for batch in progress:
                data, target = batch[0].float().to(self.device), batch[1].float().to(self.device)
                if training:
                    self.optimizer.zero_grad(set_to_none=True)
                prediction = self.model(data)
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
                    totals[key] += component.detach().item() * size
                progress.set_postfix(loss=value.item())
        return {key: value / count for key, value in totals.items()}

    def train(self, data_loader):
        history, train_losses, valid_losses = [], [], []
        for epoch in range(self.max_epoch_num):
            print("\n==== Training Epoch: %d / %d ====" % (epoch + 1, self.max_epoch_num), flush=True)
            lr = self.optimizer.param_groups[0]["lr"]
            train = self._run_epoch(data_loader["train"], True)
            valid = self._run_epoch(data_loader["valid"], False) if data_loader.get("valid") is not None else None
            self.save_model(epoch)
            if valid is not None and valid["total"] < self.min_valid_loss:
                self.min_valid_loss, self.best_epoch = valid["total"], epoch
            history.append({"epoch": epoch, "lr": lr, "train": train, "valid": valid})
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
        calculate_metrics(predictions, labels, self.config)
