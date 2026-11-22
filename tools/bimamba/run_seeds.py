"""Run configs over several seeds, then report mean +/- std (paper protocol).

From the rPPG-Toolbox root on the HPC:
    python tools/bimamba/run_seeds.py \
        --configs configs/train_configs/UBFC_UBFC_BIMAMBA_DUAL_2STAGE_PINNED.yaml \
                  configs/train_configs/UBFC_UBFC_BIMAMBA_STUDENT_SINGLE_NOKD_PINNED.yaml \
                  configs/train_configs/UBFC_UBFC_BIMAMBA_STUDENT_SINGLE_KD_PINNED.yaml \
        --teacher-config configs/train_configs/UBFC_UBFC_BIMAMBA_DUAL_2STAGE_PINNED.yaml \
        --seeds 100 200 300
    python tools/bimamba/run_seeds.py --configs ... --seeds 100 200 300 --aggregate-only
    python tools/bimamba/run_seeds.py --configs ... --dry-run          # print plan only

Behaviour
  * Seed 100 (main.py's default) keeps each config's original LOG.PATH, so the
    existing seed-100 runs are reused. Other seeds get LOG.PATH + '_seed<N>'.
  * A run is skipped if its saved_test_outputs/paper_protocol_metrics.json exists
    (resumable; use --force to rerun).
  * KD configs (MODEL.NAME BiMambaPhysKD with DISTILL.WEIGHT > 0) are paired with
    the teacher trained with the SAME seed from --teacher-config (--teacher-seed N to
    use one fixed teacher instead). Non-KD configs always run before KD configs.
    The KD student loads the checkpoint the teacher was TESTED with: last epoch, or for
    validation-selected runs (cross-dataset) the one in the teacher's checkpoint_selection.json.
  * Test outputs are read from LOG.PATH/<TEST EXP_DATA_NAME>/saved_test_outputs (differs from
    the model folder LOG.PATH/<TRAIN EXP_DATA_NAME> in cross-dataset runs).
  * Per-seed configs: configs/train_configs/seeds/<name>_seed<N>.yaml
    Logs: <run dir>/main_seed<N>.log; seed recorded in <run dir>/seed.json.
  * Summary: runs/exp/seeds/seed_summary.csv (one row per run) and
    seed_aggregate.csv (mean, std, n per config), plus a printed table.
"""
import argparse
import csv
import json
import os
import subprocess
import sys

import numpy as np
import yaml

METRICS = ('MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR')
DEFAULT_SEED = 100


def load(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def run_dir(cfg):
    """Model/log folder: LOG.PATH/<TRAIN EXP_DATA_NAME> (PreTrainedModels lives here)."""
    return os.path.join(cfg['LOG']['PATH'], cfg['TRAIN']['DATA']['EXP_DATA_NAME'])


def metrics_path(cfg):
    """config.py puts test outputs under LOG.PATH/<TEST EXP_DATA_NAME>/saved_test_outputs, which
    differs from run_dir for cross-dataset runs (same folder for intra-dataset runs)."""
    return os.path.join(cfg['LOG']['PATH'], cfg['TEST']['DATA']['EXP_DATA_NAME'], 'saved_test_outputs',
                        'paper_protocol_metrics.json')


def is_kd(cfg):
    """Needs a seed-paired teacher: output KD (DISTILL.WEIGHT) and/or feature KD (DISTILL.FEATURE.*_WEIGHT) > 0.
    Multi-teacher configs (DISTILL.TEACHER_RUNS) name their teachers themselves and need no pairing."""
    d = cfg.get('DISTILL', {}) or {}
    if d.get('TEACHER_RUNS'):
        return False
    feature = d.get('FEATURE', {}) or {}
    return cfg['MODEL']['NAME'] == 'BiMambaPhysKD' and (
        float(d.get('WEIGHT', 0)) > 0 or any(float(v) > 0 for k, v in feature.items() if k.endswith('_WEIGHT')))


def seeded(cfg, seed):
    cfg = json.loads(json.dumps(cfg))  # deep copy
    if seed != DEFAULT_SEED:
        cfg['LOG']['PATH'] = cfg['LOG']['PATH'].rstrip('/') + '_seed%d' % seed
    return cfg


def teacher_files(teacher_cfg):
    """The checkpoint the teacher was TESTED with. Last epoch when TEST.USE_LAST_EPOCH is true;
    otherwise (cross-dataset, best validation epoch) read from the teacher's
    saved_test_outputs/checkpoint_selection.json, so it can only be resolved after the teacher ran."""
    folder = os.path.join(run_dir(teacher_cfg), 'PreTrainedModels')
    checkpoint = os.path.join(folder, '%s_Epoch%d.pth' % (teacher_cfg['TRAIN']['MODEL_FILE_NAME'],
                                                        int(teacher_cfg['TRAIN']['EPOCHS']) - 1))
    if not teacher_cfg['TEST'].get('USE_LAST_EPOCH', True):
        selection = os.path.join(os.path.dirname(metrics_path(teacher_cfg)), 'checkpoint_selection.json')
        if os.path.isfile(selection):
            with open(selection) as handle:
                checkpoint = json.load(handle)['checkpoint']
        else:
            checkpoint = None  # teacher not trained yet; resolved again right before the KD run
    return checkpoint, os.path.join(folder, 'resolved_config.yaml')


def plan(args):
    base = {path: load(path) for path in args.configs}
    teacher_base = load(args.teacher_config) if args.teacher_config else None
    jobs = []
    for kd_pass in (False, True):
        for path, cfg in base.items():
            if is_kd(cfg) != kd_pass:
                continue
            if kd_pass and teacher_base is None:
                raise SystemExit('KD config %s needs --teacher-config' % path)
            for seed in args.seeds:
                job = seeded(cfg, seed)
                teacher = None
                if kd_pass:
                    t_seed = args.teacher_seed if args.teacher_seed is not None else seed
                    teacher = seeded(teacher_base, t_seed)
                name = os.path.splitext(os.path.basename(path))[0]
                jobs.append(dict(name=name, seed=seed, cfg=job, source=path, teacher=teacher,
                                 out_cfg=os.path.join(args.seed_config_dir, '%s_seed%d.yaml' % (name, seed))))
    return jobs


def execute(job, args):
    cfg = job['cfg']
    done = os.path.isfile(metrics_path(cfg))
    tag = '%s seed %d -> %s' % (job['name'], job['seed'], run_dir(cfg))
    if done and not args.force:
        print('SKIP (done)  ', tag); return
    if job.get('teacher') is not None:  # resolve now: the teacher may have been trained earlier in this run
        ckpt, tcfg = teacher_files(job['teacher'])
        cfg['DISTILL']['TEACHER_CHECKPOINT'] = ckpt or '<teacher not trained yet>'
        cfg['DISTILL']['TEACHER_CONFIG'] = tcfg
        if not args.dry_run and (ckpt is None or not os.path.isfile(ckpt)):
            raise SystemExit('Teacher checkpoint missing for %s: %s' % (tag, ckpt))
    os.makedirs(os.path.dirname(job['out_cfg']), exist_ok=True)
    with open(job['out_cfg'], 'w') as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)
    print('RUN          ', tag, '\n              config:', job['out_cfg'], flush=True)
    if job.get('teacher') is not None:
        print('              teacher:', cfg['DISTILL']['TEACHER_CHECKPOINT'], flush=True)
    if args.dry_run:
        return
    os.makedirs(run_dir(cfg), exist_ok=True)
    log = os.path.join(run_dir(cfg), 'main_seed%d.log' % job['seed'])
    env = dict(os.environ, RPPG_SEED=str(job['seed']))
    with open(log, 'w') as handle:
        code = subprocess.call([sys.executable, 'main.py', '--config_file', job['out_cfg']],
                               stdout=handle, stderr=subprocess.STDOUT, env=env)
    if code != 0 or not os.path.isfile(metrics_path(cfg)):
        raise SystemExit('FAILED (exit %d): %s; see %s' % (code, tag, log))
    with open(os.path.join(run_dir(cfg), 'seed.json'), 'w') as handle:
        json.dump(dict(seed=job['seed'], config=job['out_cfg'], source_config=job['source']), handle, indent=2)


def aggregate(jobs, out_dir):
    rows = []
    for job in jobs:
        path = metrics_path(job['cfg'])
        if not os.path.isfile(path):
            print('missing (not aggregated):', job['name'], job['seed']); continue
        with open(path) as handle:
            m = json.load(handle)
        rows.append(dict(config=job['name'], seed=job['seed'], run_dir=os.path.dirname(os.path.dirname(path)),
                         **{k: m[k] for k in METRICS}, SNR_10log10=m['SNR_10log10']))
    if not rows:
        print('Nothing to aggregate yet.'); return
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'seed_summary.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    agg = []
    for name in dict.fromkeys(r['config'] for r in rows):
        group = [r for r in rows if r['config'] == name]
        entry = dict(config=name, n=len(group), seeds=' '.join(str(r['seed']) for r in group))
        for k in METRICS:
            values = np.array([r[k] for r in group], float)
            entry[k + '_mean'] = float(values.mean())
            entry[k + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else float('nan')
        agg.append(entry)
    with open(os.path.join(out_dir, 'seed_aggregate.csv'), 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(agg[0])); writer.writeheader(); writer.writerows(agg)
    print('\nPaper protocol, mean +/- std over seeds (sample std)')
    print('%-48s %2s ' % ('config', 'n') + ' '.join('%15s' % k for k in METRICS))
    for e in agg:
        print('%-48s %2d ' % (e['config'][:48], e['n']) +
              ' '.join('%7.3f+/-%-6.3f' % (e[k + '_mean'], e[k + '_std']) for k in METRICS))
    print('\nSaved:', os.path.join(out_dir, 'seed_summary.csv'), 'and seed_aggregate.csv')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--configs', nargs='+', required=True)
    ap.add_argument('--seeds', nargs='+', type=int, default=[100, 200, 300])
    ap.add_argument('--teacher-config', help='teacher YAML used for KD configs (paired by seed)')
    ap.add_argument('--teacher-seed', type=int, help='use this teacher seed for every KD run instead of pairing')
    ap.add_argument('--seed-config-dir', default='configs/train_configs/seeds')
    ap.add_argument('--out', default='runs/exp/seeds')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true', help='rerun even if metrics exist')
    ap.add_argument('--aggregate-only', action='store_true')
    args = ap.parse_args()
    if not os.path.isfile('main.py'):
        raise SystemExit('Run from the rPPG-Toolbox root (main.py not found)')
    jobs = plan(args)
    if not args.aggregate_only:
        for job in jobs:
            execute(job, args)
    if not args.dry_run:
        aggregate(jobs, args.out)


if __name__ == '__main__':
    main()
