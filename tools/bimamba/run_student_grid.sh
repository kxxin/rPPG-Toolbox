#!/usr/bin/env bash
# Student grid (no KD, speed aug): student {OURS = single-path BiMamba, COMPACT = compact multi-temporal}
#                                 x loss {BM = BiMamba, RM = RhythmMamba, COMB = combined}.
# OURS x BM and OURS x RM already exist; this trains the 4 new cells.
# Run from the rPPG-Toolbox root:
#   bash tools/bimamba/run_student_grid.sh train CELL SETTING [seeds]   # CELL: OURS_COMB | COMPACT_BM | COMPACT_RM | COMPACT_COMB | all
#   bash tools/bimamba/run_student_grid.sh plan all all                 # dry run
#   bash tools/bimamba/run_student_grid.sh report [--seeds ...]
# SETTING: ubfc | pure | pure2ubfc | ubfc2pure | all.  Resumable.
set -euo pipefail
C=configs/train_configs

config_for() {  # cell setting
  local name
  case "$1" in
    OURS_COMB) name=BIMAMBA_STUDENT_SINGLE_NOKD_COMB ;;
    COMPACT_BM|COMPACT_RM|COMPACT_COMB) name=COMPACT6_${1#COMPACT_} ;;
    *) echo "unknown cell: $1" >&2; exit 1 ;;
  esac
  case "$2" in
    ubfc) echo "$C/UBFC_UBFC_${name}_SPEEDAUG_PINNED.yaml" ;;
    pure) echo "$C/PURE_PURE_${name}_SPEEDAUG_PINNED.yaml" ;;
    pure2ubfc) echo "$C/PURE_UBFC_${name}_SPEEDAUG_CROSS.yaml" ;;
    ubfc2pure) echo "$C/UBFC_PURE_${name}_SPEEDAUG_CROSS.yaml" ;;
    *) echo "unknown setting: $2" >&2; exit 1 ;;
  esac
}

mode=${1:-}
case "$mode" in
  train|plan)
    cell=${2:?cell}; setting=${3:?setting}; shift 3
    seeds=${*:-100 200 300}
    cells=$cell; [ "$cell" = all ] && cells="OURS_COMB COMPACT_BM COMPACT_RM COMPACT_COMB"
    settings=$setting; [ "$setting" = all ] && settings="ubfc2pure pure2ubfc ubfc pure"
    extra=""; [ "$mode" = plan ] && extra="--dry-run"
    for s in $settings; do for c in $cells; do
      echo "=== $c / $s ($mode) seeds: $seeds"
      python tools/bimamba/run_seeds.py --configs "$(config_for "$c" "$s")" --seeds $seeds \
        --out "runs/exp/seeds_grid_${c,,}_$s" $extra
    done; done ;;
  report) shift; python tools/bimamba/student_grid_report.py "$@" ;;
  *) echo "usage: $0 train|plan CELL|all SETTING|all [seeds] | report [--seeds ...]"; exit 1 ;;
esac
