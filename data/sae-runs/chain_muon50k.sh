#!/bin/bash
# Optimizer rung at 50,000 steps, so it is directly comparable to the seed and
# precision rungs rather than to a shorter run of itself.
#
# The ladder after this completes, every arm at 50k steps on the same corpus:
#   ceiling    Adam seed42 vs Adam seed123   (the seed arm running now)
#   precision  Adam bf16   vs Adam 4-bit     7.7% of matchable, 103.8x null
#   optimizer  Adam seed42 vs Muon seed42    this run
#
# Muon runs at 864 tok/s against Adam's 5,619 -- the Newton-Schulz iteration is
# applied to a 3072x16384 matrix every step -- so 50k steps is ~33 h. That cost is
# the reason for measuring it once, properly, rather than at a shorter horizon that
# would not be comparable to anything else in the table.
#
# LR 3e-3 was not swept. On a 2,000-step probe it tracked Adam's FVE to within
# 0.6 points (0.9494 vs 0.9551 at step 800) while holding 43 dead features against
# Adam's 1,307. Using it at 50k extrapolates from a compressed cosine schedule to a
# stretched one; that is a disclosed weakening of the quality match, taken in place
# of a 5.3 h four-rate sweep.
#
# TRAINER PROVENANCE. This arm uses train_sae_muon.py; the Adam arms used
# train_sae.py (sha b701c090...). The Adam code path is unchanged between them and
# the diff is recorded at data/sae-runs/trainer-muon.diff. This is the one seam in
# the optimizer rung and it is documented rather than asserted away.

set -uo pipefail
cd ~/ResearchPapers

OUT=data/sae-runs/llama3b-l14-muon-sae
log(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

log "waiting for the matched seed arm and its matcher to finish"
while pgrep -f "train_sae.py|seed_match_newcorpus.py|run_seed_arm.sh" > /dev/null; do
  sleep 300
done
log "seed arm queue is idle"

[ -f data/sae-runs/llama3b-l14-bf16-seed123-sae/checkpoint_final.npz ] \
  || { log "ABORT: seed arm did not produce a final checkpoint"; exit 1; }
log "seed arm complete"

if [ ! -f "$OUT/checkpoint_final.npz" ]; then
  mkdir -p "$OUT"
  log "=== training Muon SAE, 50,000 steps, lr 3e-3 (~33 h)"
  python3 data/sae-runs/train_sae_muon.py \
      --activations data/activations/llama3b-l14-bf16 --output "$OUT" \
      --optimizer muon --lr 3e-3 --muon-wd 0.0 \
      --dict-size 16384 --k 128 --steps 50000 --batch 2048 \
      --warmup 500 --seed 42 > "$OUT/train.log" 2>&1
  rc=$?
  log "muon arm exited rc=$rc"
  [ $rc -ne 0 ] && { log "ABORT: muon training failed"; exit $rc; }
fi

log "matching Adam vs Muon"
python3 data/sae-analysis/optimizer_match.py > /tmp/optimizer_match.log 2>&1
log "OPTIMIZER RUNG COMPLETE -- see /tmp/optimizer_match.log"

log "releasing the machine to the circuit-robustness protocol"
nohup bash data/circuit-robustness/chain_circuits_after_sae.sh > /tmp/chain_circuits2.log 2>&1 &
log "circuits chain launched"
