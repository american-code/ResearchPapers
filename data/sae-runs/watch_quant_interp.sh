#!/bin/bash
# Waits for both quant-interp SAEs to finish, then runs the matching analysis.
# Runs no training itself, so it cannot collide with the trainer.
#
# Fails loudly rather than silently: if the trainer dies, this exits non-zero
# with the reason rather than waiting forever.

set -uo pipefail
cd "$(dirname "$0")/../.."                 # -> ~/ResearchPapers

A=data/sae-runs/llama3b-l14-bf16-sae/checkpoint_final.npz
B=data/sae-runs/llama3b-l14-q4-sae/checkpoint_final.npz
DEADLINE=$(( $(date +%s) + 16*3600 ))      # 16 h ceiling; expected ~8-10 h

echo "[$(date '+%F %H:%M:%S')] watcher started; waiting for both checkpoints"

while true; do
  if [ -f "$A" ] && [ -f "$B" ]; then
    echo "[$(date '+%F %H:%M:%S')] both checkpoints present"
    break
  fi
  if ! pgrep -f "[t]rain_sae.py" > /dev/null && ! pgrep -f "[r]un_seed_stability.sh" > /dev/null; then
    echo "[$(date '+%F %H:%M:%S')] FAIL: no trainer running and checkpoints incomplete"
    echo "  seed123: $([ -f "$A" ] && echo present || echo MISSING)"
    echo "  seed456: $([ -f "$B" ] && echo present || echo MISSING)"
    echo "  last runner output:"; tail -5 data/sae-runs/quant-interp.log
    exit 1
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    echo "[$(date '+%F %H:%M:%S')] FAIL: 16 h deadline exceeded"; exit 2
  fi
  sleep 300
done

echo "[$(date '+%F %H:%M:%S')] running matching analysis"
python3 data/sae-analysis/quant_interp_match.py
rc=$?
echo "[$(date '+%F %H:%M:%S')] analysis exited rc=$rc"
exit $rc
