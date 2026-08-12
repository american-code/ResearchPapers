#!/bin/bash
DIR="$HOME/ResearchPapers/data/sae-runs/mistral-7b-layer16"
ACT="$HOME/ResearchPapers/data/activations/mistral-7b-layer16/"
TRAIN="$HOME/ResearchPapers/data/sae-runs/train_sae.py"
cd "$HOME/ResearchPapers" || exit 1
while true; do
  if ls "$DIR"/final_*.npz >/dev/null 2>&1; then
    echo "[watcher] complete"; exit 0
  fi
  if pgrep -f 'train_sae.py.*mistral' >/dev/null; then sleep 30; continue; fi
  CKPT=$(ls "$DIR"/checkpoint_step_*.npz 2>/dev/null | sort -t_ -k3 -n | tail -1)
  ARGS="--activations $ACT --output $DIR --dict-size 16384 --k 128 --lr 1e-4 --steps 50000 --batch 2048"
  [ -n "$CKPT" ] && ARGS="$ARGS --resume-from $CKPT"
  echo "[watcher] $(date -u +%FT%TZ) restarting ${CKPT:-from scratch}"
  python3 "$TRAIN" $ARGS >> "$DIR/stdout.log" 2>&1
  sleep 5
done
