"""Does the noise path matter? Exact decomposition of a trained dual-path checkpoint.

The predictor is nearest-upsample + Conv1d(k=1), i.e. linear, so for every clip
    stage2 = W*rppg + b  -  W*modeled_noise  =  rppg_part + noise_part   (exact)
    rppg_part == forward(stage=1)   (noise path switched off at inference)

Reports, on the run's own test CSV (no retraining, nothing written to the run's
PreTrainedModels):
  * paper-protocol MAE/RMSE/MAPE/Pearson/SNR for stage2 vs rppg_part (stage-1 inference)
  * noise_part / rppg_part std ratio, correlation of noise_part with GT and rppg_part,
    share of noise_part power in the GT HR band (+/-6 bpm, 1st+2nd harmonic)
  * feature-level ||modeled_noise|| / ||rppg|| and cosine(modeled_noise, rppg)
  * attention-mask focus: spatial CV, centre-mass share (uniform = 0.25), and
    mask/inverse overlap; PNG overlays for a few clips

Run on the HPC from the rPPG-Toolbox root (CUDA, real Mamba):
    python tools/bimamba/diagnose_noise_path.py --run-dir runs/exp/pinned_dual12_2stage/experiment00
    python tools/bimamba/diagnose_noise_path.py --run-dir runs/exp/pinned_student6_nokd_2stage/experiment00
Outputs go to <run-dir>/noise_path_diagnostics/ (or --out).
"""
import argparse
import csv
import glob
import json
import os
import re
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from yacs.config import CfgNode  # noqa: E402
from neural_methods.model.BiMambaPhys import BiMambaPhys  # noqa: E402
from evaluation import paper_protocol as paper  # noqa: E402


def load_run(run_dir, checkpoint=None):
    with open(os.path.join(run_dir, 'PreTrainedModels', 'resolved_config.yaml')) as handle:
        cfg = CfgNode.load_cfg(handle)
    if checkpoint is None:
        found = glob.glob(os.path.join(run_dir, 'PreTrainedModels', '*_Epoch*.pth'))
        if not found:
            raise SystemExit('No checkpoints in %s/PreTrainedModels; pass --checkpoint' % run_dir)
        checkpoint = max(found, key=lambda p: int(re.search(r'_Epoch(\d+)\.pth$', p).group(1)))
    return cfg, checkpoint


def build_model(cfg, checkpoint, device):
    m = cfg.MODEL.BIMAMBAPHYS
    if not m.DUAL_PATH:
        raise SystemExit('Not a dual-path run')
    if not getattr(m, 'NOISE_PATH', True):
        raise SystemExit('Single-path run (NOISE_PATH=False): there is no noise path to diagnose')
    model = BiMambaPhys(dim=m.DIM, depth=m.DEPTH, d_state=m.D_STATE, d_conv=m.D_CONV, expand=m.EXPAND,
                        dropout=m.DROPOUT, dual_path=True, mlp_ratio=m.MLP_RATIO,
                        drop_path_rate=m.DROP_PATH_RATE, cross_attn_heads=m.CROSS_ATTN_HEADS)
    model.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
    return model.to(device).eval()


def test_clips(cfg):
    import pandas as pd
    inputs = sorted(pd.read_csv(cfg.TEST.DATA.FILE_LIST_PATH)['input_files'].tolist())
    for path in inputs:
        name = os.path.basename(path)
        subject, chunk = name.rsplit('_input', 1)[0], int(name.rsplit('_input', 1)[1].split('.')[0])
        yield subject, chunk, path, path.replace('_input', '_label')


def band_power_share(x, hr_bpm, fs):
    """Share of x's 0.75-2.5 Hz power within +/-6 bpm of hr and 2*hr."""
    f = np.fft.rfftfreq(len(x), 1 / fs); p = np.abs(np.fft.rfft(x - x.mean())) ** 2
    band = (f >= paper.LOW_HZ) & (f <= paper.HIGH_HZ)
    hr = ((np.abs(f - hr_bpm / 60) <= 0.1) | (np.abs(f - 2 * hr_bpm / 60) <= 0.1)) & band
    return float(p[hr].sum() / max(p[band].sum(), 1e-12))


def corr(a, b):
    a, b = a - a.mean(), b - b.mean()
    return float((a * b).sum() / max(np.sqrt((a * a).sum() * (b * b).sum()), 1e-12))


def mask_stats(mask, inverse):
    """mask, inverse: [T', H', W'] for one clip."""
    t, h, w = mask.shape
    centre = np.zeros((h, w), bool); centre[h // 4:h - h // 4, w // 4:w - w // 4] = True
    m = mask.mean(0)
    return dict(mask_cv=float(mask.std(axis=(1, 2)).mean() / max(mask.mean(), 1e-12)),
                mask_centre_share=float(m[centre].sum() / m.sum()),
                inverse_centre_share=float(inverse.mean(0)[centre].sum() / inverse.mean(0).sum()),
                uniform_centre_share=float(centre.mean()),
                mask_inverse_corr=corr(mask.reshape(-1), inverse.reshape(-1)))


def save_overlay(video, mask, inverse, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = video[:, video.shape[1] // 2].transpose(1, 2, 0)
    frame = (frame - frame.min()) / max(frame.max() - frame.min(), 1e-6)
    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    ax[0].imshow(frame); ax[0].set_title('frame (standardized)')
    # Shared colour scale (mean-0.5 normalised maps): a flat mask looks flat, not auto-stretched.
    vmax = max(mask.mean(0).max(), inverse.mean(0).max(), 1.0)
    for a, data, title in ((ax[1], mask.mean(0), 'attention mask'), (ax[2], inverse.mean(0), 'inverse mask')):
        a.imshow(frame)
        im = a.imshow(data, cmap='jet', alpha=0.5, vmin=0, vmax=vmax,
                      extent=(0, frame.shape[1], frame.shape[0], 0))
        a.set_title('%s (CV %.2f)' % (title, data.std() / max(data.mean(), 1e-12)))
        fig.colorbar(im, ax=a, fraction=0.046)
    for a in ax:
        a.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    ap.add_argument('--checkpoint')
    ap.add_argument('--out')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--batch-size', type=int, default=4)
    ap.add_argument('--overlays', type=int, default=6, help='number of subjects with mask PNGs')
    args = ap.parse_args()
    cfg, checkpoint = load_run(args.run_dir, args.checkpoint)
    out = args.out or os.path.join(args.run_dir, 'noise_path_diagnostics')
    os.makedirs(out, exist_ok=True)
    model = build_model(cfg, checkpoint, args.device)
    fs = cfg.TEST.DATA.FS
    w = model.dual.predictor.weight.detach()[0, :, 0]  # [C]

    preds = {k: {} for k in ('stage2', 'rppg_part', 'noise_part')}
    labels, clip_rows, overlay_done = {}, [], set()
    clips = list(test_clips(cfg))
    print('Checkpoint:', checkpoint, '| test clips:', len(clips), flush=True)
    with torch.no_grad():
        for start in range(0, len(clips), args.batch_size):
            batch = clips[start:start + args.batch_size]
            video = torch.from_numpy(np.stack([np.load(c[2]).transpose(3, 0, 1, 2) for c in batch])).float()
            video = video.to(args.device)
            t = video.shape[2]
            stage2, aux = model.dual(video, stage=2, return_aux=True)
            stage1 = model.dual(video, stage=1)
            up = lambda x: torch.nn.functional.interpolate(x.transpose(1, 2), scale_factor=2, mode='nearest')
            noise_part = -torch.einsum('c,bct->bt', w, up(aux['modeled_noise']))[:, :t]
            rppg_part = stage2 - noise_part
            if not torch.allclose(rppg_part, stage1, atol=1e-4, rtol=1e-3):
                raise RuntimeError('Decomposition check failed: stage1 != stage2 - noise_part '
                                   '(max diff %.2e)' % (rppg_part - stage1).abs().max())
            rppg_f, noise_f = aux['rppg'], aux['modeled_noise']
            feat_ratio = (noise_f.norm(dim=(1, 2)) / rppg_f.norm(dim=(1, 2)).clamp_min(1e-12)).cpu().numpy()
            feat_cos = torch.nn.functional.cosine_similarity(noise_f.flatten(1), rppg_f.flatten(1)).cpu().numpy()
            for i, (subject, chunk, _, label_path) in enumerate(batch):
                label = np.load(label_path).astype(np.float32)
                labels.setdefault(subject, {})[chunk] = torch.from_numpy(label)
                for key, value in (('stage2', stage2), ('rppg_part', rppg_part), ('noise_part', noise_part)):
                    preds[key].setdefault(subject, {})[chunk] = value[i].float().cpu()
                n, r = noise_part[i].cpu().numpy(), rppg_part[i].cpu().numpy()
                mask = aux['mask'][i, 0].cpu().numpy(); inverse = aux['inverse_mask'][i, 0].cpu().numpy()
                row = dict(subject=subject, chunk=chunk, feature_norm_ratio=float(feat_ratio[i]),
                           feature_cosine=float(feat_cos[i]),
                           output_std_ratio=float(n.std() / max(r.std(), 1e-12)),
                           noise_part_corr_gt=corr(n, label), noise_part_corr_rppg_part=corr(n, r),
                           rppg_part_corr_gt=corr(r, label), stage2_corr_gt=corr(stage2[i].cpu().numpy(), label),
                           **mask_stats(mask, inverse))
                clip_rows.append(row)
                if subject not in overlay_done and len(overlay_done) < args.overlays:
                    save_overlay(video[i].cpu().numpy(), mask, inverse, os.path.join(out, '%s_masks.png' % subject))
                    overlay_done.add(subject)

    summaries, subject_rows = {}, []
    for key in ('stage2', 'rppg_part'):
        summary, rows = paper.evaluate(preds[key], labels, fs, cfg.TEST.DATA.PREPROCESS.LABEL_TYPE)
        summaries[key] = summary
        for r in rows:
            subject_rows.append(dict(variant=key, **r))
    # Subject-level: how much GT-HR-band power the noise path carries
    gt_hr = {r['subject']: r['hr_label'] for r in subject_rows if r['variant'] == 'stage2'}
    noise_rows = []
    for subject in sorted(preds['noise_part'], key=paper.subject_sort_key):
        n = paper.reform(preds['noise_part'][subject]); r = paper.reform(preds['rppg_part'][subject])
        noise_rows.append(dict(subject=subject, noise_hr_band_share=band_power_share(n, gt_hr[subject], fs),
                               rppg_hr_band_share=band_power_share(r, gt_hr[subject], fs),
                               output_std_ratio=float(n.std() / max(r.std(), 1e-12)),
                               noise_corr_gt=corr(n, paper.reform(labels[subject]))))

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows]))
    result = dict(
        run_dir=args.run_dir, checkpoint=checkpoint, clips=len(clip_rows),
        paper_protocol=dict(stage2_full=summaries['stage2'], stage1_noise_path_off=summaries['rppg_part']),
        noise_path=dict(feature_norm_ratio=mean(clip_rows, 'feature_norm_ratio'),
                        feature_cosine=mean(clip_rows, 'feature_cosine'),
                        output_std_ratio=mean(clip_rows, 'output_std_ratio'),
                        noise_part_corr_gt=mean(clip_rows, 'noise_part_corr_gt'),
                        noise_part_corr_rppg_part=mean(clip_rows, 'noise_part_corr_rppg_part'),
                        noise_hr_band_share=mean(noise_rows, 'noise_hr_band_share'),
                        rppg_hr_band_share=mean(noise_rows, 'rppg_hr_band_share')),
        masks=dict(mask_cv=mean(clip_rows, 'mask_cv'), mask_centre_share=mean(clip_rows, 'mask_centre_share'),
                   inverse_centre_share=mean(clip_rows, 'inverse_centre_share'),
                   uniform_centre_share=clip_rows[0]['uniform_centre_share'],
                   mask_inverse_corr=mean(clip_rows, 'mask_inverse_corr')),
        reading_guide={
            'output_std_ratio': 'noise_part std / rppg_part std; ~0 means subtraction does nothing',
            'noise_part_corr_gt': 'negative = noise path removes pulse; ~0 = removes non-pulse content',
            'noise_hr_band_share': 'share of noise_part power at GT HR harmonics; high = it carries pulse',
            'mask_centre_share': 'vs uniform_centre_share; > uniform = attention focuses on the (centred) face',
            'mask_cv': 'spatial coefficient of variation; ~0 = flat mask, inverse region is not background'})
    with open(os.path.join(out, 'noise_path_summary.json'), 'w') as handle:
        json.dump(result, handle, indent=2)
    for name, rows in (('noise_path_per_clip.csv', clip_rows), ('noise_path_per_subject.csv', noise_rows),
                       ('stage_compare_per_subject.csv', subject_rows)):
        with open(os.path.join(out, name), 'w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)

    print('\n%-26s %6s %6s %6s %8s %7s' % ('paper protocol', 'MAE', 'RMSE', 'MAPE', 'Pearson', 'SNR'))
    for label, key in (('stage 2 (full model)', 'stage2'), ('stage 1 (noise path off)', 'rppg_part')):
        s = summaries[key]
        print('%-26s %6.2f %6.2f %6.2f %8.4f %7.2f' % (label, s['MAE'], s['RMSE'], s['MAPE'], s['Pearson'], s['SNR']))
    print('\nnoise path:', json.dumps(result['noise_path'], indent=1))
    print('masks:', json.dumps(result['masks'], indent=1))
    print('\nSaved to', out)


if __name__ == '__main__':
    main()
