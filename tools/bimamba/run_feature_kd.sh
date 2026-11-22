#!/usr/bin/env bash
# PURE intra (pinned split, speed aug), everything with the RhythmMamba loss (0.2*NegPearson + 1.0*freq CE).
#   TEACHER  dual 12+12 2-stage teacher                      PURE_PURE_BIMAMBA_DUAL_2STAGE_RM_SPEEDAUG_PINNED.yaml
#   NOKD     single-path 6-block student, no KD              PURE_PURE_BIMAMBA_STUDENT_SINGLE_NOKD_RM_SPEEDAUG_PINNED.yaml
#   OUTKD    student, output KD                              ..._SINGLE_OUTKD_RM_SPEEDAUG_PINNED.yaml
#   AT       student, feature KD: attention transfer         ..._SINGLE_FKD_AT_SPEEDAUG_PINNED.yaml
#   HINT     student, feature KD: block + final hints        ..._SINGLE_FKD_HINT_SPEEDAUG_PINNED.yaml
#   ATHINT   student, feature KD: both                       ..._SINGLE_FKD_ATHINT_SPEEDAUG_PINNED.yaml
# KD students use the RhythmMamba-loss teacher of the SAME seed, so run TEACHER (for those seeds) first.
# Run from the rPPG-Toolbox root:
#   bash tools/bimamba/run_feature_kd.sh TEACHER         # seeds 100 200 300 (resumable)
#   bash tools/bimamba/run_feature_kd.sh AT 100          # chosen seeds
#   bash tools/bimamba/run_feature_kd.sh all             # TEACHER, NOKD, OUTKD, AT, HINT, ATHINT in turn
#   bash tools/bimamba/run_feature_kd.sh report          # comparison table (any time)
set -euo pipefail
C=configs/train_configs
TEACHER=$C/PURE_PURE_BIMAMBA_DUAL_2STAGE_RM_SPEEDAUG_PINNED.yaml

config_for() {
  case "$1" in
    TEACHER) echo "$TEACHER" ;;
    NOKD)    echo "$C/PURE_PURE_BIMAMBA_STUDENT_SINGLE_NOKD_RM_SPEEDAUG_PINNED.yaml" ;;
    OUTKD)   echo "$C/PURE_PURE_BIMAMBA_STUDENT_SINGLE_OUTKD_RM_SPEEDAUG_PINNED.yaml" ;;
    *)       echo "$C/PURE_PURE_BIMAMBA_STUDENT_SINGLE_FKD_$1_SPEEDAUG_PINNED.yaml" ;;
  esac
}

run_variant() {
  local v=$1; shift
  local seeds=${*:-100 200 300}
  echo "=== $v (RhythmMamba loss), seeds: $seeds"
  python tools/bimamba/run_seeds.py --configs "$(config_for "$v")" --teacher-config "$TEACHER" \
    --seeds $seeds --out "runs/exp/seeds_rm_pure_$v"
}

case "${1:-}" in
  TEACHER|NOKD|OUTKD|AT|HINT|ATHINT) v=$1; shift; run_variant "$v" "$@" ;;
  all) shift; for v in TEACHER NOKD OUTKD AT HINT ATHINT; do run_variant "$v" "$@"; done ;;
  report) shift; python tools/bimamba/feature_kd_report.py "$@" ;;
  *) echo "usage: $0 TEACHER|NOKD|OUTKD|AT|HINT|ATHINT|all [seeds...] | report"; exit 1 ;;
esac
