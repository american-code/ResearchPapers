#!/usr/bin/env bash
# run_circuit_v2.sh — IOI circuit re-derivation pipeline, dataset-v2
#
# Runs activation_patcher.py for Llama-3.1-8B and Pythia-1.4b.
# Outputs land in data/circuit-v2/.
#
# Usage (from data/ioi/ on lab-02):
#   nohup bash run_circuit_v2.sh > ../../logs/circuit-v2-$(date +%Y%m%d-%H%M%S).log 2>&1 &

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="$(dirname "$SCRIPT_DIR")/circuit-v2"
mkdir -p "$LOG_DIR"

cd "$SCRIPT_DIR"

echo "========================================================"
echo "IOI circuit v2 pipeline"
echo "Started: $(date)"
echo "Working dir: $SCRIPT_DIR"
echo "========================================================"

# Pythia-1.4b first (faster, 24L×16H = 384 passes, ~30-60 min)
echo ""
echo "--- Pythia-1.4b ---"
python3 activation_patcher.py --model pythia

echo ""
echo "Pythia done: $(date)"

# Llama-3.1-8B (slower, 32L×32H = 1024 passes, ~3-6 hours)
echo ""
echo "--- Llama-3.1-8B ---"
python3 activation_patcher.py --model llama8b

echo ""
echo "========================================================"
echo "PIPELINE COMPLETE: $(date)"
ls -lh ../../data/circuit-v2/ 2>/dev/null || ls -lh ../circuit-v2/ 2>/dev/null || true
echo "========================================================"
