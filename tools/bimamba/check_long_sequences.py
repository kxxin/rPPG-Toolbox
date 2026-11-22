"""Preflight the actual HPC CSVs and cache without preprocessing or training."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from config import get_config
from dataset.data_loader.UBFCrPPGLoader import UBFCrPPGLoader
from dataset.data_loader.ConsecutiveClipDataset import build_training_loader


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config_file', required=True)
    args = ap.parse_args()
    cfg = get_config(args)
    datasets = {}
    for split in ('TRAIN', 'TEST'):
        data = cfg[split].DATA
        if data.DO_PREPROCESS:
            raise ValueError('Preflight does not preprocess')
        ds = UBFCrPPGLoader(split.lower(), data.DATA_PATH, data, cfg.DEVICE)
        datasets[split] = ds
        expected = (data.PREPROCESS.CHUNK_LENGTH, data.PREPROCESS.RESIZE.H, data.PREPROCESS.RESIZE.W, 3)
        for video_path, label_path in zip(ds.inputs, ds.labels):
            video = np.load(video_path, mmap_mode='r', allow_pickle=False)
            label = np.load(label_path, allow_pickle=False)
            if video.shape != expected or label.shape != (expected[0],) or not np.isfinite(label).all():
                raise ValueError('Invalid cache shape or non-finite label: ' + video_path)
            del video
        print(split, len(ds), 'clips; all cache headers and labels checked')
    subject_names = lambda ds: {Path(p).name.rsplit('_input', 1)[0] for p in ds.inputs}
    if subject_names(datasets['TRAIN']) & subject_names(datasets['TEST']):
        raise ValueError('TRAIN/TEST subjects overlap')
    loader = build_training_loader(datasets['TRAIN'], cfg, None, torch.Generator().manual_seed(100))
    # Direct item access avoids spawning workers and does not consume an epoch plan.
    video, label, *_ = loader.dataset[0]
    if not np.isfinite(video).all():
        raise ValueError('Non-finite input in first joined sequence')
    print('Joined TRAIN item:', video.shape, label.shape)
    print('Original TEST item:', datasets['TEST'][0][0].shape)
    print('Preflight passed. Test clips, CSV membership and cache files are unchanged.')


if __name__ == '__main__':
    main()
