#!/usr/bin/env bash
# Speed-augmented models: intra-dataset (UBFC, PURE) and cross-dataset (UBFC->PURE, PURE->UBFC).
# Run from the rPPG-Toolbox root on the HPC.
#
#   bash tools/bimamba/run_aug_suite.sh splits                    # cross-dataset split CSVs (from existing caches)
#   bash tools/bimamba/run_aug_suite.sh train pure 100 200 300    # one group, given seeds (default 100 200 300)
#   bash tools/bimamba/run_aug_suite.sh train all                 # every group, seeds 100 200 300
#   bash tools/bimamba/run_aug_suite.sh summary all               # mean +/- std per group, no training
#   bash tools/bimamba/run_aug_suite.sh plan all                  # dry run: print what would run
#
# Groups (each: teacher dual 12+12 2-stage -> single-path no-KD -> single-path KD from that teacher):
#   ubfc | pure | ubfc2pure | pure2ubfc | all
# Intra no-KD+aug runs already done are skipped automatically (resumable). KD students use the
# teacher of the same seed; in cross runs the teacher checkpoint is its best validation epoch.
set -euo pipefail
ROOT=${ROOT:-/home/user/24020695/PreprocessedData}   # must match CACHED_PATH in the configs
SPLITS=$ROOT/splits
C=configs/train_configs

group_configs() {
  case "$1" in
    ubfc)      echo "$C/UBFC_UBFC_BIMAMBA_DUAL_2STAGE_SPEEDAUG_PINNED.yaml $C/UBFC_UBFC_BIMAMBA_STUDENT_SINGLE_NOKD_SPEEDAUG_PINNED.yaml $C/UBFC_UBFC_BIMAMBA_STUDENT_SINGLE_KD_SPEEDAUG_PINNED.yaml" ;;
    pure)      echo "$C/PURE_PURE_BIMAMBA_DUAL_2STAGE_SPEEDAUG_PINNED.yaml $C/PURE_PURE_BIMAMBA_STUDENT_SINGLE_NOKD_SPEEDAUG_PINNED.yaml $C/PURE_PURE_BIMAMBA_STUDENT_SINGLE_KD_SPEEDAUG_PINNED.yaml" ;;
    ubfc2pure) echo "$C/UBFC_PURE_BIMAMBA_DUAL_2STAGE_SPEEDAUG_CROSS.yaml $C/UBFC_PURE_BIMAMBA_STUDENT_SINGLE_NOKD_SPEEDAUG_CROSS.yaml $C/UBFC_PURE_BIMAMBA_STUDENT_SINGLE_KD_SPEEDAUG_CROSS.yaml" ;;
    pure2ubfc) echo "$C/PURE_UBFC_BIMAMBA_DUAL_2STAGE_SPEEDAUG_CROSS.yaml $C/PURE_UBFC_BIMAMBA_STUDENT_SINGLE_NOKD_SPEEDAUG_CROSS.yaml $C/PURE_UBFC_BIMAMBA_STUDENT_SINGLE_KD_SPEEDAUG_CROSS.yaml" ;;
    *) echo "unknown group: $1" >&2; exit 1 ;;
  esac
}

run_group() {  # $1 = mode (train|summary|plan), $2 = group, rest = seeds
  local mode=$1 group=$2; shift 2
  local seeds=${*:-100 200 300}
  local cfgs; cfgs=$(group_configs "$group")
  local teacher=${cfgs%% *}
  local extra=""
  [ "$mode" = summary ] && extra="--aggregate-only"
  [ "$mode" = plan ] && extra="--dry-run"
  echo "=== $group ($mode) seeds: $seeds"
  python tools/bimamba/run_seeds.py --configs $cfgs --teacher-config "$teacher" \
    --seeds $seeds --out "runs/exp/seeds_aug_$group" $extra
}

case "${1:-}" in
  splits)
    python tools/bimamba/make_pinned_split.py --dataset UBFC-rPPG --protocol cross_source --cached-path "$ROOT/experiment00" --out-dir "$SPLITS"
    python tools/bimamba/make_pinned_split.py --dataset UBFC-rPPG --protocol cross_target --cached-path "$ROOT/experiment00" --out-dir "$SPLITS"
    python tools/bimamba/make_pinned_split.py --dataset PURE --protocol cross_source --cached-path "$ROOT/PURE_128_c160_std" --out-dir "$SPLITS"
    python tools/bimamba/make_pinned_split.py --dataset PURE --protocol cross_target --cached-path "$ROOT/PURE_128_c160_std" --out-dir "$SPLITS"
    ;;
  train|summary|plan)
    mode=$1; group=${2:-}; shift 2 || true
    if [ "$group" = all ]; then
      for g in ubfc pure ubfc2pure pure2ubfc; do run_group "$mode" "$g" "$@"; done
    else
      run_group "$mode" "$group" "$@"
    fi
    ;;
  *) echo "usage: $0 splits | train|summary|plan <ubfc|pure|ubfc2pure|pure2ubfc|all> [seeds...]"; exit 1 ;;
esac
