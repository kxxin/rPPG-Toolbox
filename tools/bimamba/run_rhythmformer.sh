#!/usr/bin/env bash
# RhythmFormer baseline / candidate teacher under our protocol (RhythmMamba loss, speed aug, same trainer).
# Run from the rPPG-Toolbox root:
#   bash tools/bimamba/run_rhythmformer.sh train ubfc2pure          # 3 seeds (resumable); then ubfc | pure | pure2ubfc
#   bash tools/bimamba/run_rhythmformer.sh train-lr9e3 ubfc2pure    # ONLY if the LR 3e-4 run underfits (see README)
#   bash tools/bimamba/run_rhythmformer.sh check                    # prediction-averaging check for the teacher set
#   bash tools/bimamba/run_rhythmformer.sh report                   # baseline table incl. RhythmFormer
#   bash tools/bimamba/run_rhythmformer.sh mt9                      # 9-teacher student (only if check says so)
set -euo pipefail
C=configs/train_configs
cfg() {  # tag setting
  case "$2" in
    ubfc) echo "$C/UBFC_UBFC_$1_RM_SPEEDAUG_PINNED.yaml" ;;
    pure) echo "$C/PURE_PURE_$1_RM_SPEEDAUG_PINNED.yaml" ;;
    pure2ubfc) echo "$C/PURE_UBFC_$1_RM_SPEEDAUG_CROSS.yaml" ;;
    ubfc2pure) echo "$C/UBFC_PURE_$1_RM_SPEEDAUG_CROSS.yaml" ;;
    *) echo "setting: ubfc | pure | pure2ubfc | ubfc2pure" >&2; exit 1 ;;
  esac
}
mode=${1:-}; shift || true
case "$mode" in
  train|train-lr9e3)
    tag=RHYTHMFORMER; [ "$mode" = train-lr9e3 ] && tag=RHYTHMFORMER_LR9E3
    setting=${1:?setting}; shift
    python tools/bimamba/run_seeds.py --configs "$(cfg $tag "$setting")" --seeds ${*:-100 200 300} \
      --out "runs/exp/seeds_baseline_${tag,,}_$setting" ;;
  check) python tools/bimamba/ensemble_check.py "$@" ;;
  report) python tools/bimamba/baseline_report.py "$@" ;;
  mt9) python tools/bimamba/run_seeds.py --configs "$C/UBFC_PURE_BIMAMBA_STUDENT_SINGLE_MT9_RM_SPEEDAUG_CROSS.yaml" \
         --seeds ${*:-100 200 300} --out runs/exp/seeds_multiteacher_MT9 ;;
  *) echo "usage: $0 train|train-lr9e3 SETTING [seeds] | check [--setting S] | report | mt9 [seeds]"; exit 1 ;;
esac
