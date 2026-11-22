"""Check a preprocessed cache BEFORE deleting the raw videos.

From the rPPG-Toolbox root:
    python tools/bimamba/verify_cache.py --dataset PURE \
        --cached-path /home/user/24020695/PreprocessedData/PURE_128_c160_std \
        --raw-path /home/user/24020695/data/PURE

Checks every clip (loads all of them; a few minutes): input/label pairs exist, input
shape (chunk, H, W, 3) and dtype, label length == chunk, all values finite, clip ids
contiguous per video (0..n-1), standardized range sane; per-video clip counts vs the
raw frame counts (if --raw-path still exists); subject count. Writes
<cached-path>/cache_verification.json and exits non-zero on any problem.
Delete raw videos only after this prints "CACHE OK".
"""
import argparse
import glob
import json
import os
import re
import sys

import numpy as np

EXPECTED_SUBJECTS = {'PURE': 10, 'UBFC-rPPG': 42}


def video_and_subject(dataset, stem):
    if dataset == 'PURE':
        return stem, int(stem) // 100
    match = re.fullmatch(r'subject(\d+)', stem)
    if not match:
        raise ValueError('Unexpected cache name ' + stem)
    return stem, int(match.group(1))


def raw_frame_counts(dataset, raw_path):
    counts = {}
    if dataset == 'PURE':
        for d in glob.glob(os.path.join(raw_path, '*-*')):
            name = os.path.basename(d)
            counts[str(int(name.replace('-', '')))] = len(glob.glob(os.path.join(d, name, '*.png')))
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True, choices=['PURE', 'UBFC-rPPG', 'MMPD'])
    ap.add_argument('--cached-path', required=True)
    ap.add_argument('--raw-path')
    ap.add_argument('--chunk', type=int, default=160)
    ap.add_argument('--size', type=int, default=128)
    ap.add_argument('--expect-dtype', default='float32', help="'any' to skip the dtype check")
    args = ap.parse_args()

    inputs = sorted(glob.glob(os.path.join(args.cached_path, '*_input*.npy')))
    if not inputs:
        raise SystemExit('No clips in ' + args.cached_path)
    problems, videos, dtypes, total_bytes = [], {}, set(), 0
    value_min, value_max = np.inf, -np.inf
    for i, path in enumerate(inputs):
        stem, idx = os.path.basename(path).rsplit('_input', 1)
        idx = int(idx.split('.')[0])
        video, subject = video_and_subject(args.dataset, stem)
        videos.setdefault(video, dict(subject=subject, clips=[]))['clips'].append(idx)
        label_path = path.replace('_input', '_label')
        if not os.path.isfile(label_path):
            problems.append('missing label: ' + label_path); continue
        x = np.load(path, mmap_mode='r'); y = np.load(label_path)
        dtypes.add(str(x.dtype)); total_bytes += os.path.getsize(path) + os.path.getsize(label_path)
        if x.shape != (args.chunk, args.size, args.size, 3):
            problems.append('%s shape %s' % (path, x.shape))
        if y.shape != (args.chunk,):
            problems.append('%s label shape %s' % (label_path, y.shape))
        xa = np.asarray(x)
        if not np.isfinite(xa).all() or not np.isfinite(y).all():
            problems.append('non-finite values in ' + path)
        value_min, value_max = min(value_min, float(xa.min())), max(value_max, float(xa.max()))
        if float(np.std(y)) == 0:
            problems.append('flat label in ' + label_path)
        if (i + 1) % 200 == 0:
            print('checked %d / %d clips' % (i + 1, len(inputs)), flush=True)

    for video, info in videos.items():
        clips = sorted(info['clips'])
        if clips != list(range(len(clips))):
            problems.append('%s clip ids not contiguous: %s' % (video, clips))
    raw = raw_frame_counts(args.dataset, args.raw_path) if args.raw_path and os.path.isdir(args.raw_path) else {}
    for video, frames in raw.items():
        expected = frames // args.chunk
        got = len(videos.get(video, {}).get('clips', []))
        if got != expected:
            problems.append('%s: %d clips cached, %d expected from %d raw frames' % (video, got, expected, frames))
    subjects = sorted({v['subject'] for v in videos.values()})
    expected_subjects = EXPECTED_SUBJECTS.get(args.dataset)
    if expected_subjects and len(subjects) != expected_subjects:
        problems.append('%d subjects cached, expected %d' % (len(subjects), expected_subjects))
    if args.expect_dtype != 'any' and dtypes != {args.expect_dtype}:
        problems.append('clip dtype %s, expected %s' % (sorted(dtypes), args.expect_dtype))
    if value_max > 50 or value_min < -50:
        problems.append('input range %.1f..%.1f is not plausible for Standardized data' % (value_min, value_max))

    report = dict(dataset=args.dataset, cached_path=os.path.abspath(args.cached_path), clips=len(inputs),
                  videos=len(videos), subjects=subjects, dtypes=sorted(dtypes),
                  size_gb=round(total_bytes / 1e9, 2), input_range=[value_min, value_max],
                  clips_per_video={v: len(info['clips']) for v, info in sorted(videos.items())},
                  raw_compared=bool(raw), problems=problems, ok=not problems)
    with open(os.path.join(args.cached_path, 'cache_verification.json'), 'w') as handle:
        json.dump(report, handle, indent=2)
    print('\n%s: %d videos, %d clips, subjects %s, dtype %s, %.2f GB, input range %.2f..%.2f'
          % (args.dataset, len(videos), len(inputs), subjects, sorted(dtypes), total_bytes / 1e9, value_min, value_max))
    print('raw frame counts compared:', bool(raw))
    for p in problems[:50]:
        print('PROBLEM:', p)
    print('CACHE OK' if not problems else 'CACHE HAS %d PROBLEM(S) - do not delete the raw videos' % len(problems))
    sys.exit(0 if not problems else 1)


if __name__ == '__main__':
    main()
