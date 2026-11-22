"""Build a toolbox cache from raw videos WITHOUT training (float32 on disk).

From the rPPG-Toolbox root:
    python tools/bimamba/preprocess_only.py --config configs/preprocess_configs/PURE_PREPROCESS_128_C160_STD.yaml --check-raw
    python tools/bimamba/preprocess_only.py --config configs/preprocess_configs/PURE_PREPROCESS_128_C160_STD.yaml --workers 4

* Uses the TRAIN.DATA section of the YAML (DO_PREPROCESS must be True, BEGIN 0, END 1,
  FILE_LIST_PATH a directory) and the dataset's own toolbox loader, so the crop /
  resize / Standardized / chunking steps are exactly the toolbox's.
* Only change: clips are saved as float32 instead of the toolbox's float64 (half the
  disk; the loader casts to float32 on read anyway, so model inputs are identical).
  Done by patching BaseLoader.save/save_multi_process -- no toolbox file is edited.
  BaseLoader forces the multiprocessing 'spawn' start method, so the patch is applied
  at module top level (spawned workers re-import this script as __mp_main__) and
  switched by the inherited env var BIMAMBA_CACHE_FLOAT32.
* --workers caps the toolbox's parallel preprocessing processes (default 8 in the
  toolbox; each PURE video holds ~2-4 GB in RAM while it is processed).
* --check-raw only validates the raw folder layout and exits.
"""
import argparse
import glob
import os
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, os.getcwd())
from config import get_config  # noqa: E402
from dataset.data_loader.BaseLoader import BaseLoader  # noqa: E402

LOADERS = {'PURE': ('dataset.data_loader.PURELoader', 'PURELoader'),
           'UBFC-rPPG': ('dataset.data_loader.UBFCrPPGLoader', 'UBFCrPPGLoader'),
           'MMPD': ('dataset.data_loader.MMPDLoader', 'MMPDLoader')}


def _save_float32(self, frames_clips, bvps_clips, filename):
    os.makedirs(self.cached_path, exist_ok=True)
    inputs, labels = [], []
    for i in range(len(bvps_clips)):
        input_path = self.cached_path + os.sep + '{0}_input{1}.npy'.format(filename, i)
        label_path = self.cached_path + os.sep + '{0}_label{1}.npy'.format(filename, i)
        np.save(input_path, np.asarray(frames_clips[i], dtype=np.float32))
        np.save(label_path, np.asarray(bvps_clips[i], dtype=np.float32))
        inputs.append(input_path); labels.append(label_path)
    return inputs, labels


def _save_float32_single(self, frames_clips, bvps_clips, filename):
    inputs, labels = _save_float32(self, frames_clips, bvps_clips, filename)
    self.inputs.extend(inputs); self.labels.extend(labels)
    return len(inputs)


def apply_cache_dtype_patch():
    """Runs in the parent AND in every spawned preprocessing worker."""
    if os.environ.get('BIMAMBA_CACHE_FLOAT32') == '1':
        BaseLoader.save_multi_process = _save_float32
        BaseLoader.save = _save_float32_single


apply_cache_dtype_patch()


def check_raw_pure(path):
    dirs = sorted(d for d in glob.glob(os.path.join(path, '*-*')) if os.path.isdir(d))
    problems, frames = [], {}
    for d in dirs:
        name = os.path.basename(d)
        pngs = glob.glob(os.path.join(d, name, '*.png'))
        if not pngs:
            problems.append('%s: no PNGs in %s/%s/' % (name, name, name))
        if not os.path.isfile(os.path.join(d, name + '.json')):
            problems.append('%s: missing %s/%s.json' % (name, name, name))
        frames[name] = len(pngs)
    subjects = sorted({int(os.path.basename(d).split('-')[0]) for d in dirs})
    print('PURE raw: %d videos, subjects %s' % (len(dirs), subjects))
    print('frames per video: min %d, max %d' % (min(frames.values() or [0]), max(frames.values() or [0])))
    for p in problems:
        print('PROBLEM:', p)
    if not dirs:
        print('PROBLEM: no "SS-TT" folders under', path)
    print('Expected layout: <DATA_PATH>/01-01/01-01/*.png and <DATA_PATH>/01-01/01-01.json')
    return not problems and bool(dirs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--check-raw', action='store_true')
    ap.add_argument('--keep-float64', action='store_true', help='save exactly like the toolbox')
    args = ap.parse_args()
    config = get_config(SimpleNamespace(config_file=args.config))
    data = config.TRAIN.DATA
    if not data.DO_PREPROCESS:
        raise SystemExit('TRAIN.DATA.DO_PREPROCESS must be True in the preprocessing YAML')
    if (data.BEGIN, data.END) != (0.0, 1.0):
        raise SystemExit('Preprocess the whole dataset once: set BEGIN 0.0 and END 1.0')
    if data.DATASET not in LOADERS:
        raise SystemExit('Unsupported dataset ' + data.DATASET)
    if args.check_raw:
        if data.DATASET != 'PURE':
            raise SystemExit('--check-raw is implemented for PURE only')
        sys.exit(0 if check_raw_pure(data.DATA_PATH) else 1)
    if os.path.isdir(data.CACHED_PATH) and glob.glob(os.path.join(data.CACHED_PATH, '*_input*.npy')):
        raise SystemExit('Cache already has clips: %s (refusing to mix runs)' % data.CACHED_PATH)
    os.environ['BIMAMBA_CACHE_FLOAT32'] = '0' if args.keep_float64 else '1'  # inherited by workers
    apply_cache_dtype_patch()
    BaseLoader.multi_process_manager.__defaults__ = (args.workers,)
    module, cls = LOADERS[data.DATASET]
    loader_cls = getattr(__import__(module, fromlist=[cls]), cls)
    print('Preprocessing %s from %s -> %s (%s, %d workers)' % (
        data.DATASET, data.DATA_PATH, data.CACHED_PATH, 'float64' if args.keep_float64 else 'float32', args.workers))
    dataset = loader_cls(name='preprocess', data_path=data.DATA_PATH, config_data=data, device=config.DEVICE)
    print('Done: %d clips in %s; file list %s' % (len(dataset.inputs), data.CACHED_PATH, data.FILE_LIST_PATH))
    print('Next: python tools/bimamba/verify_cache.py --dataset %s --cached-path %s' % (data.DATASET, data.CACHED_PATH))


if __name__ == '__main__':
    main()
