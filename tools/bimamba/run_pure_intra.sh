#!/usr/bin/env bash
# PURE intra-dataset experiment (pinned split: subjects 1-6 train, 7-10 test).
# Run from the rPPG-Toolbox root on the HPC:
#   bash tools/bimamba/run_pure_intra.sh prepare        # check raw -> preprocess (float32) -> verify -> splits
#   (delete the raw PURE videos yourself ONLY after "prepare" ends with CACHE OK)
#   bash tools/bimamba/run_pure_intra.sh train          # teacher -> single no-KD -> single KD, seed 100
#   bash tools/bimamba/run_pure_intra.sh train 100 200 300   # 3 seeds (resumable)
#   bash tools/bimamba/run_pure_intra.sh summary 100 200 300
set -euo pipefail
PRE=${PRE:-configs/preprocess_configs/PURE_PREPROCESS_128_C160_STD.yaml}
C=configs/train_configs
RAW=$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['TRAIN']['DATA']['DATA_PATH'])" "$PRE")
ROOT=${ROOT:-/home/user/24020695/PreprocessedData}   # must match CACHED_PATH in the configs
CACHE=$ROOT/PURE_128_c160_std
SPLITS=$ROOT/splits
WORKERS=${WORKERS:-4}
CONFIGS="$C/PURE_PURE_BIMAMBA_DUAL_2STAGE_PINNED.yaml $C/PURE_PURE_BIMAMBA_STUDENT_SINGLE_NOKD_PINNED.yaml $C/PURE_PURE_BIMAMBA_STUDENT_SINGLE_KD_PINNED.yaml"

case "${1:-}" in
  prepare)
    python tools/bimamba/preprocess_only.py --config $PRE --check-raw
    python tools/bimamba/preprocess_only.py --config $PRE --workers "$WORKERS"
    python tools/bimamba/verify_cache.py --dataset PURE --cached-path "$CACHE" --raw-path "$RAW"
    python tools/bimamba/make_pinned_split.py --dataset PURE --protocol intra --cached-path "$CACHE" --out-dir "$SPLITS"
    python tools/bimamba/make_pinned_split.py --dataset PURE --protocol cross_target --cached-path "$CACHE" --out-dir "$SPLITS"
    python tools/bimamba/make_pinned_split.py --dataset PURE --protocol cross_source --cached-path "$CACHE" --out-dir "$SPLITS"
    du -sh "$CACHE"
    echo "PREPARE DONE. Cache verified (see $CACHE/cache_verification.json)."
    echo "Back up $RAW elsewhere before deleting it from the HPC."
    ;;
  train)
    shift; SEEDS=${*:-100}
    python tools/bimamba/run_seeds.py --configs $CONFIGS --teacher-config $C/PURE_PURE_BIMAMBA_DUAL_2STAGE_PINNED.yaml \
      --seeds $SEEDS --out runs/exp/seeds_pure
    ;;
  summary)
    shift; SEEDS=${*:-100}
    python tools/bimamba/run_seeds.py --configs $CONFIGS --teacher-config $C/PURE_PURE_BIMAMBA_DUAL_2STAGE_PINNED.yaml \
      --seeds $SEEDS --out runs/exp/seeds_pure --aggregate-only
    ;;
  *) echo "usage: $0 prepare | train [seeds...] | summary [seeds...]"; exit 1 ;;
esac
