"""Checks for the paper protocol, the pinned split tool and the manifest guard.

Run from the rPPG-Toolbox root (CPU only, no Mamba needed for this file):
    python tests/test_eval_split.py
    python tests/test_eval_split.py --rhythmmamba-post ../RhythmMamba-main/evaluation/post_process.py \
        --outputs runs/exp/bimamba_dual12_2stage/experiment00/saved_test_outputs/UBFC_UBFC_BiMambaPhys_Dual2Stage_outputs.pickle
"""
import argparse
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np  # noqa: E402
from evaluation import paper_protocol as paper  # noqa: E402

UBFC_IDS = [1, 3, 4, 5, 8, 9] + list(range(10, 19)) + [20] + list(range(22, 28)) + list(range(30, 50))


def test_paper_protocol_sanity():
    fs, t = 30, np.arange(1920) / 30
    label = np.sin(2 * np.pi * 1.5 * t)
    pred = label + 0.3 * np.random.default_rng(0).standard_normal(t.size)
    hr_label, hr_pred, snr20, snr10 = paper.metric_per_video(pred, label, fs)
    assert abs(hr_label - 90) < 0.6 and abs(hr_pred - 90) < 0.6
    assert np.isclose(snr20, 2 * snr10) and snr20 > 0
    print('PASS: paper protocol recovers 90 bpm; SNR20 = 2 x SNR10')


def test_parity(post_path, outputs):
    spec = importlib.util.spec_from_file_location('rm_post', post_path)
    rm = importlib.util.module_from_spec(spec); spec.loader.exec_module(rm)
    with open(outputs, 'rb') as handle:
        data = pickle.load(handle)
    for subject in data['predictions']:
        p, l = paper.reform(data['predictions'][subject]), paper.reform(data['labels'][subject])
        ref = rm.calculate_metric_per_video(p, l, fs=30, diff_flag=False)
        mine = paper.metric_per_video(p, l, 30, False)
        assert np.allclose(ref, mine[:3]), (subject, ref, mine)
    print('PASS: identical HR/SNR to RhythmMamba post_process on', len(data['predictions']), 'videos')


def _fake_cache(folder, names):
    for name in names:
        for k in range(3):
            np.save(os.path.join(folder, '%s_input%d.npy' % (name, k)), np.zeros(1))
            np.save(os.path.join(folder, '%s_label%d.npy' % (name, k)), np.zeros(1))


def _split(dataset, cache, out, protocol='intra'):
    subprocess.run([sys.executable, str(ROOT / 'tools' / 'bimamba' / 'make_pinned_split.py'), '--dataset', dataset,
                    '--protocol', protocol, '--cached-path', cache, '--out-dir', out], check=True,
                   stdout=subprocess.DEVNULL)
    with open(os.path.join(out, '%s_%s_manifest.json' % (dataset, protocol))) as handle:
        return json.load(handle)


def test_pinned_split():
    with tempfile.TemporaryDirectory() as tmp:
        cache, out = os.path.join(tmp, 'ubfc'), os.path.join(tmp, 'splits')
        os.makedirs(cache)
        _fake_cache(cache, ['subject%d' % i for i in UBFC_IDS])
        m = _split('UBFC-rPPG', cache, out)
        assert m['splits']['test']['subjects'] == list(range(38, 50))
        assert m['splits']['train']['subjects'] == UBFC_IDS[:30] and UBFC_IDS[29] == 37
        assert m['splits']['test']['clips'] == 36
        cross = _split('UBFC-rPPG', cache, out, 'cross_source')
        assert len(cross['splits']['train']['subjects']) == 34 and len(cross['splits']['valid']['subjects']) == 8
        pure = os.path.join(tmp, 'pure'); os.makedirs(pure)
        _fake_cache(pure, ['%d%02d' % (s, t) for s in range(1, 11) for t in (1, 2, 3, 4, 5, 6)])
        mp = _split('PURE', pure, out)
        assert mp['splits']['train']['subjects'] == [1, 2, 3, 4, 5, 6]
        assert mp['splits']['test']['subjects'] == [7, 8, 9, 10] and mp['splits']['test']['clips'] == 72
        print('PASS: UBFC test = subjects 38-49, train = first 30 (1-37); UBFC cross 34/8; PURE 1-6 / 7-10')
        return m


def test_manifest_guard(manifest_dir):
    from neural_methods.trainer.BiMambaPhysTrainer import BiMambaPhysTrainer
    from config import _C
    path = os.path.join(manifest_dir, 'UBFC-rPPG_intra_manifest.json')
    with open(path) as handle:
        m = json.load(handle)

    def loader(subjects, clips=3):
        files = ['/c/subject%d_input%d.npy' % (s, k) for s in subjects for k in range(clips)]
        return SimpleNamespace(dataset=SimpleNamespace(inputs=files))

    cfg = _C.clone(); cfg.defrost(); cfg.TRAIN.DATA.DATASET = cfg.TEST.DATA.DATASET = 'UBFC-rPPG'
    cfg.MODEL.BIMAMBAPHYS.SPLIT_MANIFESTS = [path]
    fake = SimpleNamespace(config=cfg, _subject_id=BiMambaPhysTrainer._subject_id)
    good = dict(train=loader(m['splits']['train']['subjects']), test=loader(m['splits']['test']['subjects']))
    BiMambaPhysTrainer._check_manifests(fake, good)
    assert fake.split_record['test']['subjects'] == list(range(38, 50))
    old_hpc_test = [5, 8, 11, 12, 16, 18, 23, 24, 30, 37, 40, 47]
    bad = dict(train=good['train'], test=loader(old_hpc_test))
    try:
        BiMambaPhysTrainer._check_manifests(fake, bad)
        raise AssertionError('glob-order test split was not rejected')
    except ValueError as error:
        assert 'test split differs' in str(error)
    print('PASS: manifest guard accepts the pinned split and rejects the old glob-order split')


def test_configs():
    from config import _C
    for name in ('UBFC_UBFC_BIMAMBA_DUAL_2STAGE_PINNED.yaml', 'UBFC_UBFC_BIMAMBA_STUDENT_NOKD_2STAGE_PINNED.yaml',
                 'UBFC_UBFC_BIMAMBA_KD_2STAGE_PINNED.yaml'):
        cfg = _C.clone(); cfg.defrost(); cfg.merge_from_file(str(ROOT / 'configs' / 'train_configs' / name))
        assert cfg.TRAIN.DATA.FILE_LIST_PATH.endswith('UBFC-rPPG_intra_train.csv')
        assert cfg.TEST.DATA.FILE_LIST_PATH.endswith('UBFC-rPPG_intra_test.csv')
        assert len(cfg.MODEL.BIMAMBAPHYS.SPLIT_MANIFESTS) == 1 and cfg.INFERENCE.PAPER_PROTOCOL
    print('PASS: pinned configs merge with config.py')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rhythmmamba-post', type=Path)
    ap.add_argument('--outputs', type=Path)
    args = ap.parse_args()
    test_paper_protocol_sanity()
    if args.rhythmmamba_post and args.outputs:
        test_parity(args.rhythmmamba_post, args.outputs)
    with tempfile.TemporaryDirectory() as tmp:
        cache, out = os.path.join(tmp, 'ubfc'), os.path.join(tmp, 'splits')
        os.makedirs(cache); _fake_cache(cache, ['subject%d' % i for i in UBFC_IDS]); _split('UBFC-rPPG', cache, out)
        test_pinned_split()
        test_manifest_guard(out)
    test_configs()


if __name__ == '__main__':
    main()
