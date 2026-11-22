"""Config checks for the RhythmMamba-loss suite (no GPU needed).  From the toolbox root:
    python tests/test_rm_suite.py
"""
import glob
import yaml

C = 'configs/train_configs/'
files = sorted(glob.glob(C + '*_RM_SPEEDAUG_*.yaml'))
assert len([f for f in files if 'UBFC' in f]) >= 8, files
for path in files:
    with open(path) as handle:
        c = yaml.safe_load(handle)
    m = c['MODEL']['BIMAMBAPHYS']
    assert m['LOSS']['TYPE'] == 'rhythmmamba', path
    assert c['TRAIN']['SPEED_AUG']['ENABLE'], path
    cross = path.endswith('_CROSS.yaml')
    assert c['TEST']['USE_LAST_EPOCH'] is (not cross), path
    if 'DUAL_2STAGE' in path:
        assert m['DEPTH'] == 12 and m['STAGE1_EPOCHS'] == 15 and m.get('NOISE_PATH', True), path
    else:
        assert m['DEPTH'] == 6 and m['NOISE_PATH'] is False and m['STAGE1_EPOCHS'] == 0, path
        assert c['DISTILL']['WEIGHT'] == 0.0, path
    if '_HINT_' in path:
        f = c['DISTILL']['FEATURE']
        assert (f['AT_WEIGHT'], f['HINT_WEIGHT'], f['FINAL_WEIGHT'], f['TSM_WEIGHT']) == (0, 1, 1, 0), path
        teacher = path.replace('STUDENT_SINGLE_HINT', 'DUAL_2STAGE')
        with open(teacher) as handle:
            t = yaml.safe_load(handle)
        assert c['DISTILL']['TEACHER_CHECKPOINT'].startswith(t['LOG']['PATH'] + '/'), path
    print('ok ', path.split('/')[-1])
print('all RM-suite configs consistent')
