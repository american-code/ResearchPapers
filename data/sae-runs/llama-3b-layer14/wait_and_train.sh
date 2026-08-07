#!/bin/bash
# Wait for Qwen SAE training (PID 93489) to finish, then resume Llama training from step 10000.
QWEN_PID=93489
LOGFILE=/Users/melton/ResearchPapers/data/sae-runs/llama-3b-layer14/stdout.log

echo "[$(date)] Waiting for Qwen training (PID $QWEN_PID) to finish..." >> "$LOGFILE"
while kill -0 "$QWEN_PID" 2>/dev/null; do
    sleep 60
done
echo "[$(date)] Qwen training finished. Starting Llama-3b resume from step 10000." >> "$LOGFILE"

python3 /Users/melton/ResearchPapers/data/sae-runs/train_sae.py \
    --activations /Users/melton/ResearchPapers/data/activations/llama-3b-layer14/ \
    --output      /Users/melton/ResearchPapers/data/sae-runs/llama-3b-layer14/ \
    --dict-size   16384 \
    --k           128 \
    --lr          1e-4 \
    --steps       50000 \
    --batch       2048 \
    --resume-from /Users/melton/ResearchPapers/data/sae-runs/llama-3b-layer14/checkpoint_step_010000.npz \
    >> "$LOGFILE" 2>&1

echo "[$(date)] Llama-3b SAE training complete." >> "$LOGFILE"
