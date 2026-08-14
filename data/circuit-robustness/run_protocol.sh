#!/bin/bash
# The circuit robustness protocol, end to end.
#
# For each (model, task family):
#   1. discover a circuit C on D
#   2. score F(C) on D          -- in-distribution reference
#   3. score F(C) on D-prime    -- the shifted number
#
# The comparison that matters is step 3 against step 2, NOT against the published
# single-template figures. D and D-prime have the same template count and share no
# frames, so a drop from 2 to 3 is frame dependence and nothing else. The published
# 0.663 -> 0.228 result cannot make that separation, because it changed template
# count and template identity together.
#
# Primary run is the FRAME axis across all four families. The entity and both axes
# are secondary and run for IOI only -- 4 families x 3 axes x 2 models would be 24
# discoveries, and the frame axis is what the paper's claim rests on.
#
# Must not run while lab-02 is training. Discovery is the expensive step:
# n_batches * (1 + n_layers*n_heads) forward passes for patching plus
# n_layers*n_heads*n_batches for ablation, per dataset.

set -uo pipefail
cd "$(dirname "$0")"

FAITH=score_faithfulness.py
OUT=results
mkdir -p circuits "$OUT"

log(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

if pgrep -f "train_sae.py" > /dev/null; then
  log "ABORT: train_sae.py is running -- circuit work would contend with it"
  exit 1
fi

PRIMARY="ioi factual agreement induction"
SECONDARY_SHIFTS="entity both"

run_one(){   # model family shift
  local m=$1 fam=$2 sh=$3
  local tag="$m-$fam-$sh"
  local circ="circuits/$tag.json"

  if [ ! -f "$circ" ]; then
    log "discover: $tag"
    python3 discover_circuit.py --model "$m" --dataset "$fam-$sh-D.json" \
        --out "$circ" > "$OUT/$tag.discover.log" 2>&1 \
      || { log "FAILED discovery $tag (see $OUT/$tag.discover.log)"; return 1; }
  fi

  for split in D Dprime; do
    local res="$OUT/$tag-$split.faith.json"
    [ -f "$res" ] && continue
    log "faithfulness: $tag on $split"
    python3 "$FAITH" --model "$m" \
        --dataset "$fam-$sh-$split.json" \
        --circuit-file "$circ" \
        --out "$res" > "$OUT/$tag-$split.faith.log" 2>&1 \
      || log "FAILED faithfulness $tag/$split"
  done
}

for m in llama pythia; do
  for fam in $PRIMARY; do
    run_one "$m" "$fam" frame
  done
done

for m in llama pythia; do
  for sh in $SECONDARY_SHIFTS; do
    run_one "$m" ioi "$sh"
  done
done

log "collecting"
python3 collect_results.py "$OUT" | tee "$OUT/SUMMARY.txt"
log "PROTOCOL COMPLETE -> $OUT/SUMMARY.txt"
