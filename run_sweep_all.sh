#!/bin/bash
# ── Run the full sweep across all (strength, ratio) pairs ───────────
#
# Usage:
#   # Run everything (small S in batch, large S one-at-a-time):
#   bash run_sweep_all.sh
#
#   # Run in background with logging:
#   nohup bash run_sweep_all.sh > sweep.log 2>&1 &
#
#   # Run a single (strength, ratio) pair:
#   bash run_sweep_all.sh 100 20
#
#   # Run with JAX:
#   USE_JAX=1 bash run_sweep_all.sh
#
# The script is safe to re-run: it appends to existing data files,
# so you can interrupt and resume without losing progress.
# ────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ── Configuration ───────────────────────────────────────────────────
OUT_DIR="sample_notebooks/plot_data_v2"
GEN_ARGS="0 1"     # matches original data (use "-1 1" for paper)
MEPS_FLAG=""
if [ "${USE_JAX:-0}" = "1" ]; then
    MEPS_FLAG="--use-jax"
fi

STRENGTHS=(5 25 100 300 400 500)
RATIOS=(5 20 80)

# ── Allow running a single pair from command line ───────────────────
if [ $# -ge 2 ]; then
    STRENGTHS=($1)
    RATIOS=($2)
fi

echo "========================================"
echo "Sweep v2: ${#STRENGTHS[@]} strengths x ${#RATIOS[@]} ratios"
echo "Output: $OUT_DIR"
echo "Gen args: $GEN_ARGS"
echo "JAX: ${MEPS_FLAG:-off}"
echo "========================================"

for STRENGTH in "${STRENGTHS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        echo ""
        echo "════════════════════════════════════════"
        echo "  strength=$STRENGTH, ratio=$RATIO%"
        echo "════════════════════════════════════════"

        # Small S values: run in one batch
        echo "  → Small S (5-1000)..."
        python run_sweep_v2.py $STRENGTH $RATIO \
            --s-values 5 10 25 50 100 250 500 1000 \
            --gen-args $GEN_ARGS \
            $MEPS_FLAG \
            --out-dir "$OUT_DIR"

        # Large S values: run one at a time for fault tolerance
        for S in 2000 4000 8000 12000 16000; do
            echo "  → S=$S..."
            python run_sweep_v2.py $STRENGTH $RATIO \
                --s-values $S \
                --gen-args $GEN_ARGS \
                $MEPS_FLAG \
                --out-dir "$OUT_DIR"
        done

        echo "  ✓ Done: strength=$STRENGTH, ratio=$RATIO%"
    done
done

echo ""
echo "========================================"
echo "All sweeps complete!"
echo "Verify with: python verify_sweep_data.py $OUT_DIR"
echo "========================================"
