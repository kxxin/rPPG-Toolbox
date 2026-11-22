#!/usr/bin/env bash
# Loss / augmentation ablation on UBFC-rPPG -> PURE, single-path 6-block student, no KD.
# Variant A (BiMamba loss + clip-target speed aug) = the existing cross_ubfc2pure_student6_single_nokd_speedaug runs.
#   B: RhythmMamba loss                       configs/train_configs/UBFC_PURE_BIMAMBA_STUDENT_SINGLE_NOKD_ABLATION_B.yaml
#   C: HR-balanced speed augmentation         ..._ABLATION_C.yaml
#   D: RhythmMamba loss + HR-balanced aug     ..._ABLATION_D.yaml
#   E: RhythmMamba loss + RhythmMamba aug     ..._ABLATION_E.yaml   (speed aug off)
# Run from the rPPG-Toolbox root:
#   bash tools/bimamba/run_ablation.sh B             # one variant, seeds 100 200 300 (resumable)
#   bash tools/bimamba/run_ablation.sh C 100         # one variant, chosen seeds
#   bash tools/bimamba/run_ablation.sh all           # B, C, D in turn
#   bash tools/bimamba/run_ablation.sh E             # RhythmMamba recipe (run separately; not in "all")
#   bash tools/bimamba/run_ablation.sh report        # A-E comparison table (any time)
set -euo pipefail
C=configs/train_configs

run_variant() {
  local v=$1; shift
  local seeds=${*:-100 200 300}
  echo "=== ablation $v, seeds: $seeds"
  python tools/bimamba/run_seeds.py --configs "$C/UBFC_PURE_BIMAMBA_STUDENT_SINGLE_NOKD_ABLATION_$v.yaml" \
    --seeds $seeds --out "runs/exp/seeds_ablation_$v"
}

case "${1:-}" in
  B|C|D|E) v=$1; shift; run_variant "$v" "$@" ;;
  all)   shift; for v in B C D; do run_variant "$v" "$@"; done ;;
  report) shift; python tools/bimamba/ablation_report.py "$@" ;;
  *) echo "usage: $0 B|C|D|E|all [seeds...] | report"; exit 1 ;;
esac
