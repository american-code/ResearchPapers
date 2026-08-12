#!/usr/bin/env bash
# Run this on the LOCAL machine after lab-02 training finishes.
# Rsyncs the new Qwen SAE v2 checkpoint and logs, then runs fingerprint matching
# using local activations (same Llama/Mistral SAEs as v2 baseline for comparability).
#
# Usage: bash sync_and_match_qwen_v2.sh

set -euo pipefail
cd /Users/melton/ResearchPapers

echo "=== Syncing data/qwen-sae-v2/ from lab-02 ==="
rsync -av --progress lab-02:~/ResearchPapers/data/qwen-sae-v2/ data/qwen-sae-v2/

if [[ ! -f data/qwen-sae-v2/checkpoint_final.npz ]]; then
    echo "ERROR: checkpoint_final.npz not synced — training may not be complete yet" >&2
    exit 1
fi

echo ""
echo "Training metrics:"
python -c "import json; m=json.load(open('data/qwen-sae-v2/metrics_final.json')); print(json.dumps({k: m[k] for k in ('training_elapsed_min','final_dead_count','final_dead_pct')}, indent=2))"

echo ""
echo "=== Running fingerprint matching v3 (Qwen SAE with aux revival loss) ==="
python data/sae-analysis/cross_arch_matching_v3.py 2>&1 | tee data/qwen-sae-v2/matching-stdout.log

echo ""
echo "=== All done ==="
echo "Results: data/qwen-sae-v2/matching/"
echo ""
echo "Delta vs v2 baseline:"
python -c "
import json
c = json.load(open('data/qwen-sae-v2/matching/comparison-v2-v3.json'))
for metric in ('cosine', 'pearson'):
    if metric not in c: continue
    m = c[metric]
    print(f'  [{metric}] universal: {m[\"universal_v2\"]} -> {m[\"universal_v3\"]} (delta {m[\"universal_delta\"]:+d})')
    for pair, d in m['pairwise_deltas'].items():
        print(f'    {pair}: {d[\"v2\"]} -> {d[\"v3\"]} (delta {d[\"delta\"]:+d})')
"
