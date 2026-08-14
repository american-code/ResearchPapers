#!/bin/bash
# Matched seed control on the NEW corpus.
#
# The quantization result (7.7% of matchable, Pearson) was read against a seed
# ceiling of 7.9-8.3% measured on data/activations/llama-3b-layer16 -- a DIFFERENT
# activation set from the one the quantization arms used. The gap between the two
# numbers (0.2pp) is smaller than the spread among the seed pairs themselves
# (0.33pp), so the comparison is suggestive but not controlled. This arm removes
# that: a third SAE differing from the bf16 arm ONLY in seed, on the same corpus,
# trained by the byte-identical trainer (sha b701c090...).
#
# Everything except --seed matches data/sae-runs/llama3b-l14-bf16-sae exactly.

set -uo pipefail
cd "$(dirname "$0")" 2>/dev/null || cd ~/ResearchPapers
cd ~/ResearchPapers

OUT=data/sae-runs/llama3b-l14-bf16-seed123-sae
log(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

if pgrep -f "train_sae.py" > /dev/null; then
  log "ABORT: a trainer is already running -- never two SAE jobs on one machine"; exit 1
fi

log "matched seed arm: seed 123 on data/activations/llama3b-l14-bf16 (~5.1 h)"
mkdir -p "$OUT"
python3 data/sae-runs/train_sae.py \
    --activations data/activations/llama3b-l14-bf16 --output "$OUT" \
    --dict-size 16384 --k 128 --steps 50000 --batch 2048 --lr 1e-4 \
    --warmup 500 --seed 123 > "$OUT/train.log" 2>&1
rc=$?
log "seed arm exited rc=$rc"
[ $rc -ne 0 ] && { log "ABORT: training failed"; exit $rc; }
[ -f "$OUT/checkpoint_final.npz" ] || { log "ABORT: no final checkpoint"; exit 1; }

log "matching seed42 vs seed123 on the same corpus"
python3 data/sae-analysis/seed_match_newcorpus.py > /tmp/seed_match_new.log 2>&1
log "SEED ARM COMPLETE -- see /tmp/seed_match_new.log"
