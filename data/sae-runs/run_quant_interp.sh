#!/bin/bash
# Quantization x interpretability: do SAE features survive 4-bit quantization?
#
# Trains two TopK-SAEs under identical configuration, differing in ONE variable:
# whether the activations they decompose came from bf16 weights or from 4-bit
# weights quantized from those same bf16 weights.
#
#   arm A : mlx-community/Llama-3.2-3B-bf16, layer 14
#   arm B : the same weights, locally converted with mlx_lm.convert --q-bits 4
#
# Both arms use the same corpus (data/corpus-wikitext103), the same collection
# script, the same tokenizer, and seed 42. The 4-bit arm is converted LOCALLY
# rather than downloaded, per the provenance argument in the quantization paper:
# a community upload cannot be verified as to base revision or settings.
#
# Reference points for reading the result, both from the same matching procedure:
#   same model, different seed .......... 8% of matchable agree (the ceiling)
#   different architectures ............. 0.18-0.33% (the floor)
# Near 8% => quantization preserves the representation as well as reseeding does.
# Near 0.3% => it does not.
#
# NOTE: the pre-existing bf16 SAE (data/sae-runs/llama-3b-layer14) is NOT reused.
# Its activations came from a corpus and script version that no longer reproduce
# byte-identically, so it cannot serve as a controlled arm. Both arms are fresh.
#
# Sequential throughout -- never two SAE jobs on one machine.

set -uo pipefail
cd "$(dirname "$0")/../.."                 # -> ~/ResearchPapers

CORPUS=data/corpus-wikitext103
BF16=mlx-community/Llama-3.2-3B-bf16
Q4DIR=models/llama-3.2-3b-4bit-local
ACTS_BF16=data/activations/llama3b-l14-bf16
ACTS_Q4=data/activations/llama3b-l14-q4
SAE_BF16=data/sae-runs/llama3b-l14-bf16-sae
SAE_Q4=data/sae-runs/llama3b-l14-q4-sae

log(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

log "quantization x interpretability run starting on $(hostname)"

# ── 1. bf16 activations (already collected to /tmp during verification) ───────
if [ ! -f "$ACTS_BF16/activations.bin" ]; then
  mkdir -p "$ACTS_BF16"
  if [ -f /tmp/acts-bf16-verify/activations.bin ]; then
    log "reusing verified bf16 collection from /tmp"
    cp /tmp/acts-bf16-verify/activations.bin /tmp/acts-bf16-verify/metadata.json "$ACTS_BF16/"
  else
    log "collecting bf16 activations"
    python3 data/sae-runs/collect_activations.py --model "$BF16" --layer 0.5 \
        --corpus "$CORPUS" --output "$ACTS_BF16" --token-target 500000 --seq-len 512 \
        > "$ACTS_BF16.collect.log" 2>&1 || { log "ABORT: bf16 collection failed"; exit 1; }
  fi
fi
log "bf16 activations ready: $(du -h $ACTS_BF16/activations.bin | cut -f1)"

# ── 2. local 4-bit conversion ────────────────────────────────────────────────
if [ ! -f "$Q4DIR/config.json" ]; then
  log "converting $BF16 -> 4-bit locally"
  rm -rf "$Q4DIR"                      # mlx_lm convert refuses an existing dir
  mkdir -p models
  python3 -m mlx_lm convert --hf-path "$BF16" --mlx-path "$Q4DIR" -q --q-bits 4 \
      > /tmp/convert_q4.log 2>&1 || { log "ABORT: conversion failed"; tail -5 /tmp/convert_q4.log; exit 1; }
fi
log "4-bit model ready: $(du -sh $Q4DIR | cut -f1)"

# ── 3. 4-bit activations, same corpus and script ─────────────────────────────
if [ ! -f "$ACTS_Q4/activations.bin" ]; then
  log "collecting 4-bit activations"
  python3 data/sae-runs/collect_activations.py --model "$Q4DIR" --layer 0.5 \
      --corpus "$CORPUS" --output "$ACTS_Q4" --token-target 500000 --seq-len 512 \
      > "$ACTS_Q4.collect.log" 2>&1 || { log "ABORT: 4-bit collection failed"; exit 1; }
fi
log "4-bit activations ready: $(du -h $ACTS_Q4/activations.bin | cut -f1)"

# Sanity: the two arms must differ. Identical activations would mean the
# conversion did nothing and the whole experiment is vacuous.
python3 - <<'PY'
import hashlib, sys
def h(p):
    m=hashlib.sha256()
    with open(p,"rb") as f:
        while (b:=f.read(1<<24)): m.update(b)
    return m.hexdigest()
a=h("data/activations/llama3b-l14-bf16/activations.bin")
b=h("data/activations/llama3b-l14-q4/activations.bin")
print(f"  bf16 sha {a[:16]}\n  q4   sha {b[:16]}")
if a==b:
    print("  ABORT: activations identical -- quantization had no effect"); sys.exit(1)
print("  ok: arms differ as expected")
PY
[ $? -ne 0 ] && { log "ABORT: sanity check failed"; exit 1; }

# ── 4. train both SAEs, sequentially ─────────────────────────────────────────
for arm in bf16 q4; do
  case $arm in
    bf16) A=$ACTS_BF16; O=$SAE_BF16;;
    q4)   A=$ACTS_Q4;   O=$SAE_Q4;;
  esac
  if [ -f "$O/checkpoint_final.npz" ]; then log "$arm SAE already complete, skipping"; continue; fi
  mkdir -p "$O"
  log "=== training $arm SAE -> $O (~5 h)"
  python3 data/sae-runs/train_sae.py --activations "$A" --output "$O" \
      --dict-size 16384 --k 128 --steps 50000 --batch 2048 --lr 1e-4 \
      --warmup 500 --seed 42 > "$O/train.log" 2>&1
  rc=$?
  log "$arm SAE exited rc=$rc"
  [ $rc -ne 0 ] && { log "ABORT: $arm training failed"; exit $rc; }
done

log "BOTH ARMS COMPLETE"
