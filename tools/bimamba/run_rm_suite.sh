#!/usr/bin/env bash
# RhythmMamba-loss suite (speed aug): teacher -> single-path no-KD student -> single-path HINT feature-KD student.
# Run from the rPPG-Toolbox root on the HPC. Resumable (finished seeds are skipped).
#
#   bash tools/bimamba/run_rm_suite.sh train ubfc               # UBFC intra, seeds 100 200 300
#   bash tools/bimamba/run_rm_suite.sh train pure2ubfc 100      # one group, chosen seeds
#   bash tools/bimamba/run_rm_suite.sh train all                # ubfc, pure2ubfc, ubfc2pure in turn
#   bash tools/bimamba/run_rm_suite.sh plan all                 # dry run
#   bash tools/bimamba/run_rm_suite.sh report                   # all settings incl. PURE intra
#
# Groups:
#   ubfc       UBFC intra (pinned)   teacher, NOKD, HINT
#   pure2ubfc  PURE -> UBFC-rPPG     teacher, NOKD, HINT
#   ubfc2pure  UBFC-rPPG -> PURE     teacher, HINT   (the RM no-KD student = ablation B, already run;
#                                                     run_ablation.sh B reruns it if needed)
# PURE intra was run by run_feature_kd.sh. HINT students use the teacher of the SAME seed; cross-dataset
# teachers are taken at their best validation epoch (checkpoint_selection.json), so teachers run first.
# Needs the bimamba_cross split CSVs (run_aug_suite.sh splits) - already made for the earlier cross runs.
set -euo pipefail
C=configs/train_configs

group_configs() {
  case "$1" in
    ubfc)      echo "$C/UBFC_UBFC_BIMAMBA_DUAL_2STAGE_RM_SPEEDAUG_PINNED.yaml $C/UBFC_UBFC_BIMAMBA_STUDENT_SINGLE_NOKD_RM_SPEEDAUG_PINNED.yaml $C/UBFC_UBFC_BIMAMBA_STUDENT_SINGLE_HINT_RM_SPEEDAUG_PINNED.yaml" ;;
    pure2ubfc) echo "$C/PURE_UBFC_BIMAMBA_DUAL_2STAGE_RM_SPEEDAUG_CROSS.yaml $C/PURE_UBFC_BIMAMBA_STUDENT_SINGLE_NOKD_RM_SPEEDAUG_CROSS.yaml $C/PURE_UBFC_BIMAMBA_STUDENT_SINGLE_HINT_RM_SPEEDAUG_CROSS.yaml" ;;
    ubfc2pure) echo "$C/UBFC_PURE_BIMAMBA_DUAL_2STAGE_RM_SPEEDAUG_CROSS.yaml $C/UBFC_PURE_BIMAMBA_STUDENT_SINGLE_HINT_RM_SPEEDAUG_CROSS.yaml" ;;
    *) echo "unknown group: $1 (ubfc | pure2ubfc | ubfc2pure | all)" >&2; exit 1 ;;
  esac
}

run_group() {  # mode group [seeds...]
  local mode=$1 group=$2; shift 2
  local seeds=${*:-100 200 300}
  local cfgs; cfgs=$(group_configs "$group")
  local extra=""
  [ "$mode" = plan ] && extra="--dry-run"
  echo "=== $group ($mode, RhythmMamba loss) seeds: $seeds"
  python tools/bimamba/run_seeds.py --configs $cfgs --teacher-config "${cfgs%% *}" \
    --seeds $seeds --out "runs/exp/seeds_rm_$group" $extra
}

mode=${1:-}; shift || true
case "$mode" in
  train|plan)
    group=${1:-}; shift || true
    if [ "$group" = all ]; then for g in ubfc pure2ubfc ubfc2pure; do run_group "$mode" "$g" "$@"; done
    else run_group "$mode" "$group" "$@"; fi ;;
  report) python tools/bimamba/rm_suite_report.py "$@" ;;
  *) echo "usage: $0 train|plan ubfc|pure2ubfc|ubfc2pure|all [seeds...] | report"; exit 1 ;;
esac
