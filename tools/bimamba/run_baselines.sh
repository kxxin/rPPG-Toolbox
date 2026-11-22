#!/usr/bin/env bash
# Baselines under our protocol (speed aug, RhythmMamba loss, same trainer/optimiser/evaluation as our teacher):
#   DUALPHYSMAMBA  reference DualPhysMamba reproduction (unidirectional Mamba, 12+12 blocks, 2-stage)
#   RHYTHMMAMBA    reference RhythmMamba (24 blocks, single path)
#   BIDUALPHYSMAMBA the DualPhysMamba reproduction with bidirectional Mamba (only change vs DUALPHYSMAMBA)
# Run from the rPPG-Toolbox root on the HPC. Resumable.
#
#   bash tools/bimamba/run_baselines.sh train DUALPHYSMAMBA ubfc          # one model, one setting, seeds 100 200 300
#   bash tools/bimamba/run_baselines.sh train DUALPHYSMAMBA all           # one model, every setting
#   bash tools/bimamba/run_baselines.sh train all ubfc2pure 100           # both models, chosen seeds
#   bash tools/bimamba/run_baselines.sh plan all all                      # dry run
#   bash tools/bimamba/run_baselines.sh report                            # vs our RM teacher/students + hard videos
# Settings: ubfc (UBFC intra) | pure (PURE intra) | pure2ubfc | ubfc2pure | all
set -euo pipefail
C=configs/train_configs

config_for() {  # model setting
  case "$2" in
    ubfc)      echo "$C/UBFC_UBFC_$1_RM_SPEEDAUG_PINNED.yaml" ;;
    pure)      echo "$C/PURE_PURE_$1_RM_SPEEDAUG_PINNED.yaml" ;;
    pure2ubfc) echo "$C/PURE_UBFC_$1_RM_SPEEDAUG_CROSS.yaml" ;;
    ubfc2pure) echo "$C/UBFC_PURE_$1_RM_SPEEDAUG_CROSS.yaml" ;;
    *) echo "unknown setting: $2 (ubfc | pure | pure2ubfc | ubfc2pure | all)" >&2; exit 1 ;;
  esac
}

run_one() {  # mode model setting [seeds...]
  local mode=$1 model=$2 setting=$3; shift 3
  local seeds=${*:-100 200 300}
  local extra=""; [ "$mode" = plan ] && extra="--dry-run"
  echo "=== $model / $setting ($mode, RhythmMamba loss) seeds: $seeds"
  python tools/bimamba/run_seeds.py --configs "$(config_for "$model" "$setting")" \
    --seeds $seeds --out "runs/exp/seeds_baseline_${model,,}_$setting" $extra
}

mode=${1:-}
case "$mode" in
  train|plan)
    model=${2:-}; setting=${3:-}; shift 3 || { echo "usage: $0 $mode DUALPHYSMAMBA|RHYTHMMAMBA|BIDUALPHYSMAMBA|all SETTING|all [seeds...]"; exit 1; }
    models=$model; [ "$model" = all ] && models="DUALPHYSMAMBA RHYTHMMAMBA BIDUALPHYSMAMBA"
    settings=$setting; [ "$setting" = all ] && settings="ubfc pure pure2ubfc ubfc2pure"
    for m in $models; do
      case "$m" in DUALPHYSMAMBA|RHYTHMMAMBA|BIDUALPHYSMAMBA) ;; *) echo "unknown model: $m"; exit 1 ;; esac
      for s in $settings; do run_one "$mode" "$m" "$s" "$@"; done
    done ;;
  report) shift; python tools/bimamba/baseline_report.py "$@" ;;
  *) echo "usage: $0 train|plan DUALPHYSMAMBA|RHYTHMMAMBA|BIDUALPHYSMAMBA|all ubfc|pure|pure2ubfc|ubfc2pure|all [seeds...] | report"; exit 1 ;;
esac
