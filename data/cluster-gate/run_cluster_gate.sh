#!/bin/bash
# Cluster determinism gate: does distributing a model across two machines change
# what it writes?
#
# Must not start until the quantization x interpretability SAE training finishes --
# exo would contend with the trainer for unified memory and GPU, corrupting both the
# training throughput numbers and this gate's timings.
#
#   A  mlx-single  lab-02 alone, in-process mlx_lm     (the published protocol)
#   B  exo-1node   exo serving, lab-02 only
#   C  exo-2node   exo serving, lab-02 + lab-01
#
# A vs B isolates the serving stack; B vs C isolates distribution. Llama-3.2-3B bf16
# (6.4 GB) is the probe precisely because it fits on ONE node -- the gate needs a
# model that CAN run both ways. The MoE target it clears the way for cannot.
#
# NOTE lab-01 is normally the iOS deployment box, not an ML node. Using it here is a
# deliberate one-off for the cluster arm; nothing is installed on it beyond exo,
# which was already present.

set -uo pipefail
cd "$(dirname "$0")/../.."                  # -> ~/ResearchPapers

MODEL=mlx-community/Llama-3.2-3B-bf16
RUNS=data/cluster-gate/runs
GEN=data/cluster-gate/cluster_gate_generate.py
EXO_PORT=52415

log(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

# ── guard: never run alongside SAE training ──────────────────────────────────
if pgrep -f "train_sae.py" > /dev/null; then
  log "ABORT: train_sae.py is still running -- the gate would contend with it"
  exit 1
fi

# ── prompts ──────────────────────────────────────────────────────────────────
if [ ! -f /tmp/evalplus_prompts/humaneval.json ]; then
  log "dumping EvalPlus prompts"
  ~/evalplus-env/bin/python data/slm-benchmark/dump_prompts.py /tmp/evalplus_prompts \
    || { log "ABORT: prompt dump failed"; exit 1; }
fi

# ── arm A: raw mlx_lm, single node ───────────────────────────────────────────
if [ ! -f "$RUNS/A-mlx-single/raw.jsonl" ]; then
  log "=== arm A: mlx_lm, lab-02 alone"
  python3 "$GEN" --backend mlx --model "$MODEL" --tag A-mlx-single \
      --out "$RUNS" --nodes 1 2>&1 | tee "$RUNS-A.log" || exit 1
fi

# ── arm B: exo, one node ─────────────────────────────────────────────────────
# Start exo with only this host in the cluster. Wait for the model to report ready
# rather than sleeping a fixed interval -- download and shard placement are variable.
start_exo(){
  log "starting exo ($1)"
  ( cd ~/exo && nohup uv run exo > /tmp/exo-gate-$1.log 2>&1 & echo $! > /tmp/exo-gate.pid )
  for _ in $(seq 1 120); do
    curl -sf "http://localhost:$EXO_PORT/v1/models" > /dev/null && { log "exo up"; return 0; }
    sleep 5
  done
  log "ABORT: exo did not come up in 10 min; see /tmp/exo-gate-$1.log"
  return 1
}
stop_exo(){ [ -f /tmp/exo-gate.pid ] && kill "$(cat /tmp/exo-gate.pid)" 2>/dev/null; sleep 5; }

if [ ! -f "$RUNS/B-exo-1node/raw.jsonl" ]; then
  log "=== arm B: exo, 1 node"
  log "    lab-01 must NOT be running exo for this arm -- verify before continuing"
  ssh lab-01 'pgrep -f "exo" > /dev/null' && { log "ABORT: exo running on lab-01"; exit 1; }
  start_exo 1node || exit 1
  python3 "$GEN" --backend exo --model "$MODEL" --endpoint "http://localhost:$EXO_PORT" \
      --tag B-exo-1node --out "$RUNS" --nodes 1 2>&1 | tee "$RUNS-B.log"
  rc=${PIPESTATUS[0]}; stop_exo
  [ "$rc" -ne 0 ] && { log "ABORT: arm B failed"; exit 1; }
fi

# ── arm C: exo, two nodes ────────────────────────────────────────────────────
if [ ! -f "$RUNS/C-exo-2node/raw.jsonl" ]; then
  log "=== arm C: exo, 2 nodes (lab-02 + lab-01)"
  ssh lab-01 'cd ~/exo && nohup uv run exo > /tmp/exo-gate-worker.log 2>&1 &' \
    || { log "ABORT: could not start exo on lab-01"; exit 1; }
  start_exo 2node || exit 1
  # Confirm the cluster actually has two nodes. A silently-single-node "cluster"
  # would produce a spurious PASS -- the most dangerous outcome this gate can have.
  N=$(curl -s "http://localhost:$EXO_PORT/v1/topology" | python3 -c \
      'import json,sys; d=json.load(sys.stdin); print(len(d.get("nodes", d)))' 2>/dev/null || echo "?")
  log "cluster reports $N node(s)"
  if [ "$N" != "2" ]; then
    log "ABORT: expected 2 nodes, got $N -- a single-node run would falsely pass the gate"
    stop_exo; ssh lab-01 'pkill -f exo'; exit 1
  fi
  python3 "$GEN" --backend exo --model "$MODEL" --endpoint "http://localhost:$EXO_PORT" \
      --tag C-exo-2node --out "$RUNS" --nodes 2 2>&1 | tee "$RUNS-C.log"
  rc=${PIPESTATUS[0]}; stop_exo; ssh lab-01 'pkill -f exo'
  [ "$rc" -ne 0 ] && { log "ABORT: arm C failed"; exit 1; }
fi

# ── score and compare ────────────────────────────────────────────────────────
for arm in A-mlx-single B-exo-1node C-exo-2node; do
  if [ ! -f "$RUNS/$arm/eval_results.json" ]; then
    log "scoring $arm"
    ~/evalplus-env/bin/evalplus.evaluate --dataset humaneval \
        --samples "$RUNS/$arm/humaneval-samples.jsonl" > "$RUNS/$arm/evalplus.log" 2>&1
  fi
done

log "=== comparison"
python3 data/cluster-gate/compare_arms.py "$RUNS" A-mlx-single B-exo-1node C-exo-2node \
    | tee data/cluster-gate/GATE-RESULT.txt
log "GATE COMPLETE -> data/cluster-gate/GATE-RESULT.txt"
