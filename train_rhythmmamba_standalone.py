"""
train_rhythmmamba_standalone.py
===============================
Standalone trainer for RhythmMamba on data produced by
`preprocess_ubfc_phys_standalone.py`. It reuses RhythmMamba's own model and loss
(imported from the repo), so results match the toolbox, but it reads the .npy
chunks directly — no YAML/config plumbing required.

IMPORTANT — data representation:
    RhythmMamba's Fusion_Stem does its OWN temporal differencing internally, so
    it expects **Standardized** frames, not DiffNormalized. Re-run preprocessing
    with:  --data_type Standardized --label_type Standardized
    and set this script's --label_type to match (controls diff_flag in the loss).

Requirements: run inside the RhythmMamba conda env (torch + mamba_ssm + causal-conv1d,
i.e. CUDA). Run FROM the RhythmMamba-main root so `neural_methods` / `evaluation`
import correctly.

------------------------------------------------------------------------------
USAGE (from RhythmMamba-main root):

    python /path/to/train_rhythmmamba_standalone.py \
        --cached_path "/home/racha/PreprocessedData" \
        --fs 35 --label_type Standardized \
        --epochs 30 --batch_size 4 --lr 3e-4 \
        --device cuda:0 --model_dir ./runs/ubfcphys_rhythmmamba --do_test

Subject-independent split: chunks are grouped by subject (the 's<n>' prefix of
the filename) so no participant appears in more than one split.
------------------------------------------------------------------------------
"""
import argparse
import glob
import os
import re
import random

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# --- RhythmMamba repo imports (run from the repo root) -----------------------
from neural_methods.model.RhythmMamba import RhythmMamba
from neural_methods.loss.TorchLossComputer import Hybrid_Loss
from evaluation.post_process import calculate_hr


# ----------------------------------------------------------------------------
# Dataset over preprocessed .npy chunks
# ----------------------------------------------------------------------------
def _diff_normalize(data):
    """Frame-to-frame normalized difference (for on-the-fly use if inputs were stored as raw uint8)."""
    diff = (data[1:] - data[:-1]) / (data[1:] + data[:-1] + 1e-7)
    std = np.std(diff)
    diff = diff / std if std > 0 else diff * 0.0
    out = np.concatenate([diff, np.zeros((1, *data.shape[1:]), dtype=np.float32)], axis=0)
    out[np.isnan(out)] = 0
    return out.astype(np.float32)


class NpyChunkDataset(Dataset):
    """Loads <index>_input<k>.npy / <index>_label<k>.npy and returns NDCHW tensors.

    If the input was stored as raw uint8 (--store_uint8 in preprocessing), it is
    normalized ON THE FLY here using `input_norm` (Standardized for RhythmMamba),
    so disk stays small with no change to what the model sees. If the input is
    already a float (pre-transformed at preprocessing), it is used as-is.
    """

    def __init__(self, input_files, input_norm="Standardized"):
        self.inputs = sorted(input_files)
        self.labels = [f.replace("input", "label") for f in self.inputs]
        self.input_norm = input_norm

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        raw = np.load(self.inputs[i])                          # (D, H, W, 3)
        if raw.dtype == np.uint8:                              # raw stored -> normalize now
            data = raw.astype(np.float32)
            if self.input_norm == "Standardized":
                data = (data - data.mean()) / (data.std() + 1e-7)
            elif self.input_norm == "DiffNormalized":
                data = _diff_normalize(data)
        else:
            data = raw.astype(np.float32)                      # already transformed at preprocessing
        label = np.load(self.labels[i]).astype(np.float32)    # (D,)
        data = np.transpose(data, (0, 3, 1, 2))               # NDCHW per-sample: (D, C, H, W)
        fname = os.path.basename(self.inputs[i])
        m = re.search(r"input(\d+)\.npy$", fname)
        chunk_id = int(m.group(1)) if m else 0
        subj = fname.split("_")[0]                            # e.g. "s1"
        return data, label, f"{subj}_{fname.split('_input')[0]}", chunk_id


def subject_of(path):
    return os.path.basename(path).split("_")[0]   # 's1', 's2', ...


def split_by_subject(cached_path, val_frac, test_frac, seed):
    inputs = glob.glob(os.path.join(cached_path, "*_input*.npy"))
    if not inputs:
        raise SystemExit(f"No *_input*.npy found in {cached_path}")
    subjects = sorted({subject_of(p) for p in inputs})
    rng = random.Random(seed)
    rng.shuffle(subjects)
    n = len(subjects)
    n_test = max(1, int(round(test_frac * n))) if test_frac > 0 else 0
    n_val = max(1, int(round(val_frac * n))) if val_frac > 0 else 0
    test_s = set(subjects[:n_test])
    val_s = set(subjects[n_test:n_test + n_val])
    train_s = set(subjects[n_test + n_val:])

    def pick(sset):
        return [p for p in inputs if subject_of(p) in sset]

    print(f"Subjects: {n} total -> train={len(train_s)}, val={len(val_s)}, test={len(test_s)}")
    return pick(train_s), pick(val_s), pick(test_s)


# ----------------------------------------------------------------------------
# Train / validate / test
# ----------------------------------------------------------------------------
def normalize_pred(p):
    return (p - torch.mean(p, axis=-1).view(-1, 1)) / (torch.std(p, axis=-1).view(-1, 1) + 1e-7)


def run_epoch_train(model, loader, criterion, optimizer, scheduler, device, fs, diff_flag, epoch):
    model.train()
    tbar = tqdm(loader, ncols=80)
    for batch in tbar:
        tbar.set_description(f"Train epoch {epoch}")
        data, labels = batch[0].float().to(device), batch[1].float().to(device)
        N = data.shape[0]
        optimizer.zero_grad()
        pred = normalize_pred(model(data))
        loss = 0.0
        for ib in range(N):
            loss = loss + criterion(pred[ib], labels[ib], epoch, fs, diff_flag)
        loss = loss / N
        loss.backward()
        optimizer.step()
        scheduler.step()
        tbar.set_postfix(loss=float(loss.item()))


@torch.no_grad()
def run_valid(model, loader, criterion, device, fs, diff_flag, epochs):
    model.eval()
    losses = []
    for batch in tqdm(loader, ncols=80, desc="Valid"):
        data, labels = batch[0].float().to(device), batch[1].float().to(device)
        N = data.shape[0]
        pred = normalize_pred(model(data))
        for ib in range(N):
            losses.append(float(criterion(pred[ib], labels[ib], epochs, fs, diff_flag).item()))
    return float(np.mean(losses)) if losses else float("inf")


@torch.no_grad()
def run_test_hr(model, loader, device, fs, diff_flag):
    """Per-chunk HR MAE/RMSE/Pearson against the BVP label (FFT method)."""
    model.eval()
    hr_pred_all, hr_gt_all = [], []
    for batch in tqdm(loader, ncols=80, desc="Test"):
        data = batch[0].float().to(device)
        labels = batch[1].float().numpy()
        pred = normalize_pred(model(data)).cpu().numpy()
        for ib in range(pred.shape[0]):
            try:
                hp, hg = calculate_hr(pred[ib], labels[ib], fs=fs, diff_flag=bool(diff_flag))
                hr_pred_all.append(hp); hr_gt_all.append(hg)
            except Exception:
                continue
    if not hr_pred_all:
        print("Test: no valid HR estimates."); return
    hp = np.array(hr_pred_all); hg = np.array(hr_gt_all)
    mae = np.mean(np.abs(hp - hg))
    rmse = np.sqrt(np.mean((hp - hg) ** 2))
    mape = np.mean(np.abs((hp - hg) / hg)) * 100
    corr = np.corrcoef(hp, hg)[0, 1] if len(hp) > 1 else float("nan")
    print(f"\nTest HR (per chunk, n={len(hp)}):  MAE={mae:.2f}  RMSE={rmse:.2f}  "
          f"MAPE={mape:.2f}%  Pearson={corr:.3f}")


def main():
    ap = argparse.ArgumentParser(description="Standalone RhythmMamba trainer over preprocessed .npy chunks.")
    ap.add_argument("--cached_path", required=True, help="Folder with *_input*.npy / *_label*.npy")
    ap.add_argument("--fs", type=float, default=35.0, help="Sampling rate (UBFC-PHYS ~35).")
    ap.add_argument("--label_type", choices=["Standardized", "DiffNormalized"], default="Standardized",
                    help="Must match how labels were preprocessed; sets diff_flag in the loss.")
    ap.add_argument("--input_norm", choices=["Standardized", "DiffNormalized", "none"], default="Standardized",
                    help="Applied ON THE FLY only when inputs were stored as raw uint8 "
                         "(--store_uint8). Use 'Standardized' for RhythmMamba. Ignored for float inputs.")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val_frac", type=float, default=0.2)
    ap.add_argument("--test_frac", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--model_dir", default="./runs/rhythmmamba_standalone")
    ap.add_argument("--model_name", default="RhythmMamba")
    ap.add_argument("--use_last_epoch", action="store_true",
                    help="Skip validation-based model selection; test with the last epoch.")
    ap.add_argument("--do_test", action="store_true", help="Run HR evaluation on the test split after training.")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. RhythmMamba (mamba_ssm) needs a GPU; this will likely fail on CPU.")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    device = torch.device(args.device)
    diff_flag = 1 if args.label_type == "DiffNormalized" else 0
    os.makedirs(args.model_dir, exist_ok=True)

    tr, va, te = split_by_subject(args.cached_path, args.val_frac, args.test_frac, args.seed)
    train_loader = DataLoader(NpyChunkDataset(tr, args.input_norm), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, drop_last=True)
    valid_loader = DataLoader(NpyChunkDataset(va, args.input_norm), batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers) if va else None
    test_loader = DataLoader(NpyChunkDataset(te, args.input_norm), batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers) if te else None
    print(f"Chunks: train={len(tr)}, val={len(va)}, test={len(te)}")
    if len(tr) == 0:
        raise SystemExit("No training chunks. Check --cached_path / split fractions.")

    model = RhythmMamba().to(device)
    model = torch.nn.DataParallel(model, device_ids=[device.index or 0])
    criterion = Hybrid_Loss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    steps = max(1, len(train_loader))
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr,
                                                    epochs=args.epochs, steps_per_epoch=steps)

    best_loss, best_epoch = None, 0
    for epoch in range(args.epochs):
        print(f"\n==== Epoch {epoch} ====")
        run_epoch_train(model, train_loader, criterion, optimizer, scheduler, device, args.fs, diff_flag, epoch)
        ckpt = os.path.join(args.model_dir, f"{args.model_name}_Epoch{epoch}.pth")
        torch.save(model.state_dict(), ckpt)
        print("Saved:", ckpt)

        if valid_loader is not None and not args.use_last_epoch:
            vloss = run_valid(model, valid_loader, criterion, device, args.fs, diff_flag, args.epochs)
            print(f"validation loss: {vloss:.4f}")
            if best_loss is None or vloss < best_loss:
                best_loss, best_epoch = vloss, epoch
                print(f"Update best model! Best epoch: {best_epoch}")

    # choose model for testing
    if args.use_last_epoch or valid_loader is None:
        best_ckpt = os.path.join(args.model_dir, f"{args.model_name}_Epoch{args.epochs - 1}.pth")
    else:
        best_ckpt = os.path.join(args.model_dir, f"{args.model_name}_Epoch{best_epoch}.pth")
        print(f"\nBest epoch: {best_epoch}  (val loss {best_loss:.4f})")

    if args.do_test and test_loader is not None:
        print(f"\nLoading {best_ckpt} for test.")
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
        run_test_hr(model, test_loader, device, args.fs, diff_flag)

    print("\nDone.")


if __name__ == "__main__":
    main()
