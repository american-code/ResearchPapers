#!/bin/bash
# Wait for every SAE job on lab-02 to finish, smoke-test the circuit pipeline against
# real weights, then run the full robustness protocol.
#
# The smoke test is not ceremony. Nothing in data/circuit-robustness/ has ever seen
# model weights -- it was written and verified structurally while lab-02 was busy --
# and this chain fires unattended in the middle of the night. A shape or attribute
# error in the Pythia attention path would otherwise be discovered after burning
# hours, or worse, produce numbers nobody checks. So: cheapest possible run of each
# distinct code path first (both architectures, both scripts), abort loudly on
# failure, and only then commit to the full sweep.
#
# Launch detached:
#   nohup bash data/circuit-robustness/chain_circuits_after_sae.sh &> /tmp/chain_circuits.log &

set -uo pipefail
cd "$(dirname "$0")"

SMOKE=/tmp/circuit-smoke
log(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

BUSY="run_quant_interp.sh|watch_quant_interp.sh|quant_interp_match.py|chain_muon_after_quant.sh|run_muon_arm.sh|train_sae.py"

log "waiting for the SAE queue to drain (quantization arms, matcher, Muon arm)"
while pgrep -f "$BUSY" > /dev/null; do
  sleep 300
done
log "lab-02 is free"

# ── smoke: cheapest run of each distinct code path ───────────────────────────
mkdir -p "$SMOKE"
smoke_fail(){ log "ABORT: smoke test failed -- $1"; log "see $SMOKE/"; exit 1; }

log "smoke 1/3: pythia discovery (24x16 heads, 25 examples)"
python3 discover_circuit.py --model pythia --dataset ioi-frame-D.json \
    --out "$SMOKE/pythia.json" --limit 25 --top-k 5 > "$SMOKE/pythia.log" 2>&1 \
  || smoke_fail "pythia discovery; tail: $(tail -3 "$SMOKE/pythia.log")"

log "smoke 2/3: llama discovery (28x24 heads, 25 examples)"
python3 discover_circuit.py --model llama --dataset ioi-frame-D.json \
    --out "$SMOKE/llama.json" --limit 25 --top-k 5 > "$SMOKE/llama.log" 2>&1 \
  || smoke_fail "llama discovery; tail: $(tail -3 "$SMOKE/llama.log")"

log "smoke 3/3: scoring (pythia circuit on D-prime)"
python3 score_faithfulness.py --model pythia --dataset ioi-frame-Dprime.json \
    --circuit-file "$SMOKE/pythia.json" --out "$SMOKE/score.json" \
    --limit 25 > "$SMOKE/score.log" 2>&1 \
  || smoke_fail "scoring; tail: $(tail -3 "$SMOKE/score.log")"

# A smoke run that "succeeds" while the model fails the task tells us nothing, and
# would let the full sweep produce 24 uninterpretable rows. Check the sign.
python3 - "$SMOKE/llama.json" "$SMOKE/pythia.json" <<'PY' || exit 1
import json, sys
for p in sys.argv[1:]:
    m = json.load(open(p))["meta"]
    c, k = m["clean_logit_diff"], m["corrupt_logit_diff"]
    print(f"  {m['model_key']:7} clean {c:7.3f}  corrupt {k:7.3f}  gap {c-k:7.3f}")
    if c - k <= 0:
        sys.exit(f"ABORT: {m['model_key']} shows no clean-over-corrupt gap on IOI; "
                 "the task is not being performed and every downstream number is noise.")
PY
log "smoke passed"

# ── full protocol ────────────────────────────────────────────────────────────
log "starting full protocol"
bash run_protocol.sh
rc=$?
log "protocol exited rc=$rc"
[ -f results/SUMMARY.txt ] && { log "--- SUMMARY ---"; cat results/SUMMARY.txt; }
