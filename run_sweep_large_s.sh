#!/bin/bash
# ── Accumulate trials for large S values one at a time ──────────────
#
# Use this to incrementally build up trial counts for large systems
# that are too expensive to batch. Each run appends to the existing
# .npz file.
#
# Usage:
#   # Run 1 trial of S=16000 for all (strength, ratio) pairs:
#   bash run_sweep_large_s.sh 16000
#
#   # Run 3 trials of S=4000:
#   bash run_sweep_large_s.sh 4000 3
#
#   # Accumulate by running repeatedly:
#   for i in $(seq 1 10); do bash run_sweep_large_s.sh 16000; done
#
#   # Run in parallel across multiple terminals/screens:
#   # Terminal 1: bash run_sweep_large_s.sh 16000
#   # Terminal 2: bash run_sweep_large_s.sh 12000
#   # Terminal 3: bash run_sweep_large_s.sh 8000
#
#   # With JAX:
#   USE_JAX=1 bash run_sweep_large_s.sh 16000
# ────────────────────────────────────────────────────────────────────

set -e

S_VALUE=${1:?"Usage: run_sweep_large_s.sh <S> [N_trials]"}
N_TRIALS=${2:-1}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="sample_notebooks/plot_data_v2"
GEN_ARGS="0 1"
MEPS_FLAG=""
if [ "${USE_JAX:-0}" = "1" ]; then
    MEPS_FLAG="--use-jax"
fi

STRENGTHS=(5 25 100 300 400 500)
RATIOS=(5 20 80)

echo "Running S=$S_VALUE with N=$N_TRIALS trials"
echo "────────────────────────────────────────"

for STRENGTH in "${STRENGTHS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        echo "  strength=$STRENGTH, ratio=$RATIO%..."
        python run_sweep_v2.py $STRENGTH $RATIO \
            --s-values $S_VALUE \
            --n-trials $N_TRIALS \
            --gen-args $GEN_ARGS \
            $MEPS_FLAG \
            --out-dir "$OUT_DIR"
    done
done

echo ""
echo "Done. Check progress: python verify_sweep_data.py $OUT_DIR"
