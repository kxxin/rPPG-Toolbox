#!/usr/bin/env bash
# Multi-teacher output KD on UBFC-rPPG -> PURE (single-path 6-block student, RhythmMamba loss, speed aug).
#   MT6     teachers: DualPhysMamba repro x3 seeds + RhythmMamba x3 seeds
#   MT3DPM  teachers: DualPhysMamba repro x3 seeds
# The teachers are the existing baseline runs (bimamba_baselines); they must exist under runs/exp.
# Run from the rPPG-Toolbox root:
#   bash tools/bimamba/run_multiteacher.sh check          # teachers present + which checkpoint each uses
#   bash tools/bimamba/run_multiteacher.sh MT6            # seeds 100 200 300 (resumable)
#   bash tools/bimamba/run_multiteacher.sh MT3DPM 100
#   bash tools/bimamba/run_multiteacher.sh all            # MT6 then MT3DPM
#   bash tools/bimamba/run_multiteacher.sh report
set -euo pipefail
C=configs/train_configs

run_variant() {
  local v=$1; shift
  local seeds=${*:-100 200 300}
  echo "=== multi-teacher KD $v, seeds: $seeds"
  python tools/bimamba/run_seeds.py --configs "$C/UBFC_PURE_BIMAMBA_STUDENT_SINGLE_${v}_RM_SPEEDAUG_CROSS.yaml" \
    --seeds $seeds --out "runs/exp/seeds_multiteacher_$v"
}

case "${1:-}" in
  check) python tools/bimamba/multiteacher_report.py --check ;;
  MT6|MT3DPM) v=$1; shift; run_variant "$v" "$@" ;;
  all) shift; for v in MT6 MT3DPM; do run_variant "$v" "$@"; done ;;
  report) shift; python tools/bimamba/multiteacher_report.py "$@" ;;
  *) echo "usage: $0 check | MT6|MT3DPM|all [seeds...] | report"; exit 1 ;;
esac
