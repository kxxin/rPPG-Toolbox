"""Write explicit, numerically sorted subject splits as toolbox FILE_LIST_PATH CSVs.

Why: the loaders build splits from an unsorted glob, so BEGIN/END=0.72 gives a
filesystem-dependent test set (on the HPC: subjects 5, 8, 11, ..., 47). These
CSVs pin the published protocols from existing caches; no re-preprocessing.

Protocols (DualPhysMamba Sec. 4.3/4.4, RhythmMamba):
  UBFC-rPPG intra : first 30 subjects (numeric order) train, last 12 test
  PURE intra      : subjects 1-6 train, 7-10 test
  MMPD intra      : 7:1:2 train/valid/test by sorted subject
  cross (source)  : 80 % train / 20 % valid by sorted subject; target = all subjects

Usage (run on the HPC, from the rPPG-Toolbox root):
  python tools/bimamba/make_pinned_split.py --dataset UBFC-rPPG \
      --cached-path /home/user/24020695/PreprocessedData/experiment00 \
      --out-dir /home/user/24020695/PreprocessedData/splits
  python tools/bimamba/make_pinned_split.py --dataset PURE --cached-path <PURE cache> --out-dir ...
  python tools/bimamba/make_pinned_split.py --dataset UBFC-rPPG --protocol cross_source ...
  python tools/bimamba/make_pinned_split.py --dataset PURE --protocol cross_target ...
"""
import argparse
import csv
import glob
import hashlib
import json
import os
import re

EXPECTED_SUBJECTS = {'UBFC-rPPG': 42, 'PURE': 10}


def subject_of(dataset, filename):
    """Cache filename (e.g. subject38_input3.npy, 1003_input0.npy) -> integer subject id."""
    stem = filename.rsplit('_input', 1)[0]
    if dataset in ('UBFC-rPPG', 'MMPD'):
        match = re.fullmatch(r'subject(\d+)', stem)
        if not match:
            raise ValueError('Unexpected %s cache name: %s' % (dataset, filename))
        return int(match.group(1))
    if dataset == 'PURE':
        # PURELoader saves index = int('SS-TT'.replace('-', '')) -> 101 ... 1006.
        if not stem.isdigit():
            raise ValueError('Unexpected PURE cache name: %s' % filename)
        return int(stem) // 100
    raise ValueError('Unsupported dataset ' + dataset)


def split_subjects(dataset, protocol, subjects):
    n = len(subjects)
    if protocol == 'intra':
        if dataset == 'UBFC-rPPG':
            return dict(train=subjects[:30], test=subjects[30:])
        if dataset == 'PURE':
            return dict(train=[s for s in subjects if s <= 6], test=[s for s in subjects if s >= 7])
        if dataset == 'MMPD':
            a, b = round(n * 0.7), round(n * 0.8)
            return dict(train=subjects[:a], valid=subjects[a:b], test=subjects[b:])
    if protocol == 'cross_source':
        a = round(n * 0.8)
        return dict(train=subjects[:a], valid=subjects[a:])
    if protocol == 'cross_target':
        return dict(test=list(subjects))
    raise ValueError('Unsupported protocol %s for %s' % (protocol, dataset))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['UBFC-rPPG', 'PURE', 'MMPD'])
    ap.add_argument('--protocol', default='intra', choices=['intra', 'cross_source', 'cross_target'])
    ap.add_argument('--cached-path', required=True, help='folder holding *_input*.npy / *_label*.npy')
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--allow-unexpected-count', action='store_true')
    args = ap.parse_args()

    inputs = sorted(glob.glob(os.path.join(args.cached_path, '*_input*.npy')))
    if not inputs:
        raise SystemExit('No *_input*.npy in ' + args.cached_path)
    clips = {}
    for path in inputs:
        label = path.replace('_input', '_label')
        if not os.path.isfile(label):
            raise SystemExit('Missing label for ' + path)
        clips.setdefault(subject_of(args.dataset, os.path.basename(path)), []).append(os.path.abspath(path))
    subjects = sorted(clips)
    expected = EXPECTED_SUBJECTS.get(args.dataset)
    if expected and len(subjects) != expected and not args.allow_unexpected_count:
        raise SystemExit('Found %d %s subjects in cache, expected %d. Is the cache complete? '
                         '(--allow-unexpected-count to override)' % (len(subjects), args.dataset, expected))

    parts = split_subjects(args.dataset, args.protocol, subjects)
    os.makedirs(args.out_dir, exist_ok=True)
    manifest = dict(dataset=args.dataset, protocol=args.protocol, cached_path=os.path.abspath(args.cached_path),
                    ordering='numeric subject id, ascending', splits={})
    for name, members in parts.items():
        files = sorted(f for s in members for f in clips[s])
        csv_path = os.path.join(args.out_dir, '%s_%s_%s.csv' % (args.dataset, args.protocol, name))
        with open(csv_path, 'w', newline='') as handle:  # same layout as BaseLoader.build_file_list
            writer = csv.writer(handle)
            writer.writerow(['', 'input_files'])
            for i, f in enumerate(files):
                writer.writerow([i, f])
        with open(csv_path, 'rb') as handle:
            digest = hashlib.sha256(handle.read()).hexdigest()
        manifest['splits'][name] = dict(csv=os.path.abspath(csv_path), subjects=members,
                                        clips=len(files), sha256=digest)
        print('%-6s %3d subjects %4d clips -> %s' % (name, len(members), len(files), csv_path))
        print('       subjects:', members)
    names = list(parts)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if set(parts[a]) & set(parts[b]):
                raise SystemExit('Subject overlap between %s and %s' % (a, b))
    manifest_path = os.path.join(args.out_dir, '%s_%s_manifest.json' % (args.dataset, args.protocol))
    with open(manifest_path, 'w') as handle:
        json.dump(manifest, handle, indent=2)
    print('Manifest:', manifest_path)


if __name__ == '__main__':
    main()
