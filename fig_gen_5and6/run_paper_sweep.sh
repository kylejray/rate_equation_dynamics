#!/bin/bash
# Run the paper-reproduction sweep (gen-args -1 1, α in kT directly).
#
# Usage:
#   run_paper_sweep.sh <gpu_id> <strengths_csv> <label> [S_VALUES...]
#
# Examples:
#   # First pass: S up to 4000 across a few strengths on GPU 1
#   bash run_paper_sweep.sh 1 0.25,1,3 PA
#
#   # Extend existing data with larger S (appends; safe to re-run)
#   bash run_paper_sweep.sh 1 0.25,1,3 PA_large 8000 12000 16000
#
# Always writes to fig_gen_5and6/data_fig6/

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

GPU_ID=${1:?"gpu_id required"}
STRENGTHS_CSV=${2:?"strengths_csv required (comma-separated, e.g. 0.25,1,3)"}
LABEL=${3:?"label required"}
shift 3 || true

if [ $# -eq 0 ]; then
    S_VALUES=(5 10 25 50 100 250 500 1000 2000 4000)
else
    S_VALUES=("$@")
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
# Preallocate a large fraction of the visible GPU so other users can see
# that this GPU is in use (preallocate-false makes our memory footprint
# invisibly small). 0.85 leaves headroom for system buffers.
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.50
export LD_LIBRARY_PATH=""
export PATH="$REPO_ROOT/.venv/bin:$PATH"

IFS=',' read -r -a STRENGTHS <<< "$STRENGTHS_CSV"
RATIOS=(5 20 80)

echo "==== Paper sweep $LABEL | GPU $GPU_ID | strengths=${STRENGTHS[*]} | S=${S_VALUES[*]} | start $(date -Is) ===="

for STRENGTH in "${STRENGTHS[@]}"; do
    for RATIO in "${RATIOS[@]}"; do
        echo ""
        echo "---- [$LABEL] strength=$STRENGTH ratio=$RATIO%  @ $(date -Is)"
        python run_sweep_v2.py "$STRENGTH" "$RATIO" \
            --s-values "${S_VALUES[@]}" \
            --gen-args -1 1 \
            --use-jax \
            --out-dir fig_gen_5and6/data_fig6 \
            || echo "  (pair failed, continuing)"
    done
done

echo ""
echo "==== Paper sweep $LABEL DONE $(date -Is) ===="
