#!/usr/bin/env bash
# Retrain Qwen-3B-layer18 SAE with auxiliary dead-latent revival loss (lambda=0.1).
# Expected runtime: 6-10h on lab-02 depending on GPU throughput.
#
# After this finishes, run sync_and_match_qwen_v2.sh on the LOCAL machine to
# rsync the checkpoint and run fingerprint matching (which uses local activations
# so it can compare apples-to-apples with the v2 baseline that also ran locally).

set -euo pipefail
cd ~/ResearchPapers

echo "=== Qwen SAE v2: aux revival loss retraining ==="
echo "Start: $(date)"
echo "Host:  $(hostname)"

ACTS=data/activations/qwen-3b-layer18
OUTPUT=data/qwen-sae-v2

mkdir -p "$OUTPUT"

python3 data/sae-runs/train_sae_aux.py \
    --activations "$ACTS" \
    --output      "$OUTPUT" \
    --dict-size   16384 \
    --k           128 \
    --lr          1e-4 \
    --steps       50000 \
    --batch       2048 \
    --warmup      500 \
    --seed        42 \
    --lambda-aux  0.1 \
    --k-aux       512 \
    2>&1 | tee "$OUTPUT/stdout.log"

echo ""
echo "=== Training complete: $(date) ==="

if [[ ! -f "$OUTPUT/checkpoint_final.npz" ]]; then
    echo "ERROR: checkpoint_final.npz not found — training may have failed" >&2
    exit 1
fi

echo "Final metrics:"
python3 -c "import json; m=json.load(open('$OUTPUT/metrics_final.json')); print(json.dumps({k: m[k] for k in ('training_elapsed_min','final_dead_count','final_dead_pct')}, indent=2))"
echo ""
echo "Next: run sync_and_match_qwen_v2.sh on the local machine."
