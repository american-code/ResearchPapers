#!/bin/bash
# Wait for the quantization x interpretability run (and its matcher) to finish, then
# install the optimizer-capable trainer and run the Muon arm.
#
# Why the trainer is installed HERE rather than now: the bf16 arm is already training
# under the current train_sae.py, and the 4-bit arm has not launched yet. Copying the
# new file in immediately would leave the two quantization arms trained by two
# different versions of the trainer. The Adam code path is unchanged -- but "the diff
# does not touch this path" is an argument, and byte-identical provenance is a fact.
# The quantization arms get the fact; only the Muon arm sees the new file.
#
# Launch detached:  nohup bash data/sae-runs/chain_muon_after_quant.sh &> /tmp/chain_muon.log &

set -uo pipefail
cd "$(dirname "$0")/../.."                  # -> ~/ResearchPapers

NEW_TRAINER=/tmp/train_sae_muon.py
log(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

[ -f "$NEW_TRAINER" ] || { log "ABORT: staged trainer $NEW_TRAINER missing"; exit 1; }

log "waiting for the quantization run to finish"
while pgrep -f "run_quant_interp.sh|watch_quant_interp.sh|train_sae.py|quant_interp_match.py" > /dev/null; do
  sleep 300
done
log "quantization pipeline is idle"

# Refuse to proceed on a failed run -- a Muon arm is only interpretable next to a
# completed precision pair.
for d in data/sae-runs/llama3b-l14-bf16-sae data/sae-runs/llama3b-l14-q4-sae; do
  [ -f "$d/checkpoint_final.npz" ] || { log "ABORT: $d did not finish"; exit 1; }
done
log "both quantization arms complete"

cp data/sae-runs/train_sae.py data/sae-runs/train_sae.py.pre-muon
cp "$NEW_TRAINER" data/sae-runs/train_sae.py
log "installed optimizer-capable trainer (previous saved as train_sae.py.pre-muon)"
diff data/sae-runs/train_sae.py.pre-muon data/sae-runs/train_sae.py \
     > data/sae-runs/trainer-muon.diff
log "diff recorded at data/sae-runs/trainer-muon.diff"

log "starting Muon arm"
bash data/sae-runs/run_muon_arm.sh
log "chain finished rc=$?"
