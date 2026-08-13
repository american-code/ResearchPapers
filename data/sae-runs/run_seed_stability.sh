#!/bin/bash
# Seed-stability control for the cross-model matching method.
#
# Trains two TopK-SAEs on IDENTICAL activations from the SAME model, differing
# only in --seed (which sets both weight init and batch order). Matching those
# two dictionaries gives the ceiling of the matching procedure: what it returns
# when feature sharing is guaranteed by construction.
#
# Without this number the cross-model result (1 shared triple against a null of
# 0.15) cannot be read -- we do not know what the detector does on a case where
# the answer should be yes.
#
# Config is identical to the paper's Llama SAE (data/sae-runs/llama-3b-layer14):
# dict 16384, k 128, 50k steps, batch 2048, lr 1e-4, warmup 500, same activations.
# Only the seed differs. The paper used seed 42; these use 123 and 456.
#
# Runs SEQUENTIALLY -- never two SAE jobs on one machine.

set -uo pipefail
cd "$(dirname "$0")/../.."                 # -> ~/ResearchPapers
ACTS=data/activations/llama-3b-layer16     # dir name says 16; metadata says layer 14

echo "[$(date '+%F %H:%M:%S')] seed-stability run starting on $(hostname)"
echo "  activations: $ACTS"
echo "  expected: ~4.1 h per seed, ~8.2 h total"

for SEED in 123 456; do
  OUT=data/sae-runs/llama-3b-layer14-seed${SEED}
  mkdir -p "$OUT"
  if [ -f "$OUT/checkpoint_final.npz" ]; then
    echo "[$(date '+%F %H:%M:%S')] seed $SEED already complete, skipping"
    continue
  fi
  echo "[$(date '+%F %H:%M:%S')] === training seed $SEED -> $OUT"
  python3 data/sae-runs/train_sae.py \
      --activations "$ACTS" \
      --output      "$OUT" \
      --dict-size 16384 --k 128 --steps 50000 \
      --batch 2048 --lr 1e-4 --warmup 500 \
      --seed "$SEED" > "${OUT}/train.log" 2>&1
  rc=$?
  echo "[$(date '+%F %H:%M:%S')] seed $SEED exited rc=$rc"
  if [ $rc -ne 0 ]; then
    echo "ABORT: seed $SEED failed; not starting the next run. See ${OUT}/train.log"
    exit $rc
  fi
done

echo "[$(date '+%F %H:%M:%S')] BOTH SEEDS COMPLETE"
