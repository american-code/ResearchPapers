#!/bin/bash
# Watches Qwen training and resumes from latest checkpoint if it dies
DIR="$HOME/ResearchPapers/data/sae-runs/qwen-3b-layer18"
ACTIVATIONS="$HOME/ResearchPapers/data/activations/qwen-3b-layer18/"
TRAIN="$HOME/ResearchPapers/data/sae-runs/train_sae.py"

while true; do
    # Find latest checkpoint
    CKPT=$(ls "$DIR"/checkpoint_step_*.npz 2>/dev/null | sort | tail -1)
    [ -z "$CKPT" ] && CKPT=""

    # Check if final exists — done
    if ls "$DIR"/final_*.npz 2>/dev/null | grep -q .; then
        echo "[watcher] Qwen complete — final checkpoint found."
        exit 0
    fi

    # Check if training is already running
    if pgrep -f "train_sae.py.*qwen" > /dev/null; then
        sleep 30
        continue
    fi

    echo "[watcher] No Qwen training process found — starting/resuming..."
    ARGS="--activations $ACTIVATIONS --output $DIR"
    [ -n "$CKPT" ] && ARGS="$ARGS --resume-from $CKPT"
    python3 "$TRAIN" $ARGS >> "$DIR/stdout.log" 2>&1
    echo "[watcher] Process exited — checking for completion..."
    sleep 5
done
