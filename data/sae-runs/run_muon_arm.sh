#!/bin/bash
# Optimizer arm for the perturbation ladder: does Adam vs Muon change the dictionary?
#
# Third rung of the ladder, sharing one instrument with the other two:
#   seed        same model, different seed ....... 8%   of matchable agree (ceiling)
#   precision   bf16 vs local 4-bit .............. running
#   optimizer   Adam vs Muon ..................... this script
#   floor       different architectures .......... 0.18-0.33%
#
# Trains on data/activations/llama3b-l14-bf16 -- the SAME activations the bf16 arm
# used -- so the model, layer, corpus, tokenizer, dictionary size, K, batch, steps
# and seed are all held fixed and the update rule is what moves.
#
# ── On matching quality rather than learning rate ────────────────────────────
# Adam and Muon have natural learning rates two orders of magnitude apart (Muon
# additionally rescales internally by sqrt(rows/cols) in apply_single). Holding --lr
# fixed across the two would NOT be a clean control: it would pin a nuisance
# parameter and let reconstruction quality float, so a low agreement score could
# just mean "Muon was undertrained" and would be uninterpretable.
#
# So the arms are matched on RECONSTRUCTION QUALITY instead. A short probe finds the
# Muon LR whose FVE matches or beats Adam's at the same step, and the full run uses
# it. This is also the control the sae-comparison result argues for: if quality
# metrics cannot distinguish dictionaries that disagree about 92% of their content,
# then equal-quality dictionaries are precisely the right thing to compare.
#
# Muon weight decay is forced to 0.0 (MLX defaults it to 0.01) because the Adam arm
# applies none; leaving it on would make this two variables.
#
# Sequential. Must not start until the quantization arms finish.

set -uo pipefail
cd "$(dirname "$0")/../.."                  # -> ~/ResearchPapers

ACTS=data/activations/llama3b-l14-bf16
ADAM_SAE=data/sae-runs/llama3b-l14-bf16-sae
MUON_SAE=data/sae-runs/llama3b-l14-muon-sae
PROBE_DIR=data/sae-runs/muon-lr-probe
PROBE_STEPS=2000
LRS="3e-3 1e-2 3e-2 1e-1"

log(){ echo "[$(date '+%F %H:%M:%S')] $*"; }

if pgrep -f "train_sae.py" > /dev/null; then
  log "ABORT: train_sae.py already running -- never two SAE jobs on one machine"
  exit 1
fi
[ -f "$ADAM_SAE/checkpoint_final.npz" ] || { log "ABORT: Adam arm not finished"; exit 1; }

# ── probe ────────────────────────────────────────────────────────────────────
# The Adam reference is a fresh 2,000-step probe, NOT step 2,000 of the full Adam
# run. Those are not the same thing: the full run warms up over 500 steps and decays
# cosine over 50,000, so at step 2,000 it is still near peak LR, whereas a 2,000-step
# probe has already annealed to the cosine floor. Comparing the two would have handed
# Muon a completed anneal against an Adam still mid-schedule and quietly flattered it.
# Every probe therefore runs the identical schedule shape.
mkdir -p "$PROBE_DIR"
if [ ! -f "$PROBE_DIR/adam-ref/checkpoint_final.npz" ]; then
  log "probe: adam lr=1e-4 for $PROBE_STEPS steps (reference)"
  python3 data/sae-runs/train_sae.py --activations "$ACTS" --output "$PROBE_DIR/adam-ref" \
      --optimizer adam --lr 1e-4 \
      --dict-size 16384 --k 128 --steps "$PROBE_STEPS" --batch 2048 \
      --warmup 200 --seed 42 --ckpt-interval 999999 > "$PROBE_DIR/adam-ref.log" 2>&1 \
    || { log "ABORT: reference probe failed"; exit 1; }
fi
REF=$(python3 -c "
import json
last = [json.loads(l) for l in open('$PROBE_DIR/adam-ref/training.jsonl') if l.strip()][-1]
print(f'{last[\"fve\"]:.4f}')")
log "Adam reference FVE after $PROBE_STEPS steps: $REF"

for lr in $LRS; do
  out="$PROBE_DIR/lr$lr"
  [ -f "$out/checkpoint_final.npz" ] && { log "probe lr=$lr done, skipping"; continue; }
  log "probe: muon lr=$lr for $PROBE_STEPS steps"
  python3 data/sae-runs/train_sae.py --activations "$ACTS" --output "$out" \
      --optimizer muon --lr "$lr" --muon-wd 0.0 \
      --dict-size 16384 --k 128 --steps "$PROBE_STEPS" --batch 2048 \
      --warmup 200 --seed 42 --ckpt-interval 999999 > "$out.log" 2>&1 \
    || { log "probe lr=$lr FAILED, see $out.log"; }
done

BEST=$(python3 - "$PROBE_DIR" "$REF" <<'PY'
import json, sys
from pathlib import Path
root, ref = Path(sys.argv[1]), float(sys.argv[2])
rows = []
for d in sorted(root.glob("lr*")):
    log = d / "training.jsonl"
    if not log.exists():
        continue
    last = None
    for line in log.read_text().splitlines():
        if line.strip():
            last = json.loads(line)
    if last:
        rows.append((d.name[2:], last["fve"], last["loss"], last.get("dead_5k")))
for lr, fve, loss, dead in rows:
    mark = "  <- >= Adam" if fve >= ref else ""
    print(f"  lr={lr:<6} FVE={fve:.4f}  loss={loss:.4f}  dead={dead}{mark}", file=sys.stderr)
if not rows:
    print(""); sys.exit(0)
# Prefer the lowest LR that reaches Adam's quality -- among LRs that clear the bar,
# the smallest is the least likely to have got there by taking large, unstable steps.
ok = [r for r in rows if r[1] >= ref]
print(sorted(ok, key=lambda r: float(r[0]))[0][0] if ok
      else max(rows, key=lambda r: r[1])[0])
PY
)
[ -z "$BEST" ] && { log "ABORT: no probe results"; exit 1; }
log "selected Muon lr=$BEST"

# ── full run ─────────────────────────────────────────────────────────────────
if [ ! -f "$MUON_SAE/checkpoint_final.npz" ]; then
  mkdir -p "$MUON_SAE"
  log "=== training Muon SAE -> $MUON_SAE (~6 h)"
  python3 data/sae-runs/train_sae.py --activations "$ACTS" --output "$MUON_SAE" \
      --optimizer muon --lr "$BEST" --muon-wd 0.0 \
      --dict-size 16384 --k 128 --steps 50000 --batch 2048 \
      --warmup 500 --seed 42 > "$MUON_SAE/train.log" 2>&1
  rc=$?
  log "Muon SAE exited rc=$rc"
  [ $rc -ne 0 ] && { log "ABORT: Muon training failed"; exit $rc; }
fi

log "MUON ARM COMPLETE -- run quant_interp_match.py over the Adam/Muon pair next"
