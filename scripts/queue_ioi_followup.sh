#!/bin/bash
# Follow-up pass for ioi-suite-20260730-184210.
#
# That run left three stages failed and two needing a re-run at the correct
# operating point:
#
#   FAILED  pythia-path-patching   — used model.model.layers; GPT-NeoX exposes
#                                    blocks at model.layers. Fixed.
#   FAILED  patching-{llama,pythia}-v2 — read dataset["meta"]["template"], which the
#                                    multi-template v2 dataset does not have. The
#                                    scripts now prefer per-example corrupt_prompt.
#   REDO    faithfulness-{llama,pythia}-v1 — ran with add_special_tokens=False; the
#                                    rest of the harness uses True, so clean LD came
#                                    out 4.77 against the paper's 5.64. Needed so the
#                                    v1-vs-v2 faithfulness comparison is not
#                                    confounded by tokenization.
#
# Same retrieval design as the first pass: ship each artifact as it is produced,
# stage a copy in .outbox/<job>/, write the sentinel last.
#
# Usage: ./scripts/queue_ioi_followup.sh [--dry-run]

set -euo pipefail

REMOTE=lab-02
RUSER=lab-02
RDIR="/Users/${RUSER}/ResearchPapers"
LOCAL="/Users/melton/ResearchPapers"
JOB="ioi-followup-$(date +%Y%m%d-%H%M%S)"
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }

say "Preflight"
ssh -o ConnectTimeout=10 "$REMOTE" true
RUNNING=$(ssh "$REMOTE" "pgrep -af '[t]rain_sae|[c]ollect_activations|[r]un_patching|[r]un_ablation|[r]un_circuit|[r]un_path_patching' || true")
[[ -n "$RUNNING" ]] && { echo "REFUSING — job already running:"; echo "$RUNNING"; exit 1; }
say "box is free"

[[ $DRY -eq 1 ]] && { say "dry run"; exit 0; }

say "Pushing corrected scripts"
ssh "$REMOTE" "mkdir -p ${RDIR}/.outbox/${JOB}"
rsync -az \
  "$LOCAL"/data/ioi/{run_patching_llama3b.py,run_patching_pythia1b.py,run_path_patching_pythia1b.py,run_circuit_faithfulness.py,dataset-v2.json} \
  "$REMOTE:${RDIR}/data/ioi/"

say "Building wrapper for $JOB"
ssh "$REMOTE" "mkdir -p ~/.pacer-done && cat > /tmp/${JOB}.sh" <<WRAPPER
#!/bin/bash
set -o pipefail
cd ${RDIR}
LOG=${RDIR}/data/ioi/${JOB}.log
exec >> "\$LOG" 2>&1
echo "=== ${JOB} started \$(date -u +%FT%TZ) ==="
OUTBOX=${RDIR}/.outbox/${JOB}

ship() {
  for f in "\$@"; do
    if [ -f "\$f" ]; then cp "\$f" "\$OUTBOX/" && echo "  staged \$(basename \$f)"
    else echo "  ship: MISSING \$f"; fi
  done
}
stage() {
  local name="\$1"; shift
  echo ""; echo "--- STAGE \$name  \$(date -u +%FT%TZ) ---"
  "\$@"; local rc=\$?
  echo "--- STAGE \$name exit=\$rc ---"; return \$rc
}
FAILED=""

# 1. Pythia path patching (v1 dataset) — was broken by the layer-access bug.
stage pythia-path-patching python3 data/ioi/run_path_patching_pythia1b.py \\
  || FAILED="\$FAILED pythia-path-patching"
ship ${RDIR}/data/ioi/path-patching-pythia1b-real.json

# 2. v1 faithfulness at the correct operating point (BOS), both models.
stage faithfulness-llama-v1-bos python3 data/ioi/run_circuit_faithfulness.py --model llama \\
  || FAILED="\$FAILED faithfulness-llama-v1-bos"
mv -f ${RDIR}/data/ioi/faithfulness-llama3b.json ${RDIR}/data/ioi/faithfulness-llama3b-v1bos.json 2>/dev/null
ship ${RDIR}/data/ioi/faithfulness-llama3b-v1bos.json

stage faithfulness-pythia-v1-bos python3 data/ioi/run_circuit_faithfulness.py --model pythia \\
  || FAILED="\$FAILED faithfulness-pythia-v1-bos"
mv -f ${RDIR}/data/ioi/faithfulness-pythia1b.json ${RDIR}/data/ioi/faithfulness-pythia1b-v1bos.json 2>/dev/null
ship ${RDIR}/data/ioi/faithfulness-pythia1b-v1bos.json

# 3. The multi-template patching sweeps. The long ones (~40 min Llama, ~10 min Pythia).
stage patching-llama3b-v2 python3 data/ioi/run_patching_llama3b.py \\
  --dataset dataset-v2.json --out patching-llama3b-v2.json \\
  || FAILED="\$FAILED patching-llama3b-v2"
ship ${RDIR}/data/ioi/patching-llama3b-v2.json

stage patching-pythia1b-v2 python3 data/ioi/run_patching_pythia1b.py \\
  --dataset dataset-v2.json --out patching-pythia1b-v2.json \\
  || FAILED="\$FAILED patching-pythia1b-v2"
ship ${RDIR}/data/ioi/patching-pythia1b-v2.json

ship "\$LOG"
LANDED=\$(ls -1 "\$OUTBOX" 2>/dev/null | tr '\\n' ' ')
if [ -z "\$FAILED" ]; then
  echo "{\"success\":true,\"job\":\"${JOB}\",\"artifacts\":\"\$LANDED\"}" > ~/.pacer-done/${JOB}.json
else
  echo "{\"success\":false,\"job\":\"${JOB}\",\"failed_stages\":\"\$FAILED\",\"artifacts\":\"\$LANDED\"}" > ~/.pacer-done/${JOB}.json
fi
echo "=== ${JOB} finished \$(date -u +%FT%TZ) failed:'\$FAILED' ==="
WRAPPER

say "Launching detached"
PID=$(ssh "$REMOTE" "chmod +x /tmp/${JOB}.sh && nohup /tmp/${JOB}.sh > /tmp/${JOB}.out 2>&1 & echo \$!")
LOGF="${RDIR}/data/ioi/${JOB}.log"

say "Verifying real progress"
for i in $(seq 1 36); do
  sleep 10
  ssh "$REMOTE" "kill -0 $PID 2>/dev/null" || {
    echo "PROCESS DIED:"; ssh "$REMOTE" "tail -30 $LOGF"; exit 1; }
  OUT=$(ssh "$REMOTE" "awk '/STAGE pythia-path-patching/,0' $LOGF 2>/dev/null | grep -vE 'Fetching|warn|urllib3'" || true)
  echo "$OUT" | grep -qiE "traceback|error:" && { echo "ERROR:"; echo "$OUT" | tail -20; exit 1; }
  echo "$OUT" | grep -qE "Edges to test|n_layers=|edge " && { say "progress confirmed"; break; }
done

echo
echo "REMOTE STARTED: ${JOB} pid=${PID} host=${REMOTE}"
echo "Pull results:  ./scripts/pull_ioi_results.sh ${JOB}"
echo "Monitor:       ssh ${REMOTE} 'tail -f ${LOGF}'"
