#!/bin/bash
# Queue the outstanding IOI experiments on lab-02.
#
# Fire-and-verify: this script pushes code, launches a detached wrapper, waits only
# until the first stage reports real progress, then exits. The wrapper runs for
# hours on its own.
#
# The critical difference from previous remote runs: the wrapper RSYNCS ITS OUTPUTS
# BACK to this machine after every stage, and only then writes its completion
# sentinel. The Jul-29 SAE runs completed on lab-02 and were never retrieved
# because nothing in the pipeline ever copied artifacts home. See CORRECTIONS.md.
#
# Usage:  ./scripts/queue_ioi_lab02.sh [--dry-run]

set -euo pipefail

REMOTE=lab-02
RUSER=lab-02
RDIR="/Users/${RUSER}/ResearchPapers"
LOCAL="/Users/melton/ResearchPapers"
JOB="ioi-suite-$(date +%Y%m%d-%H%M%S)"
BACK_HOST="${BACK_HOST:-$(hostname -s)}"   # for the reverse rsync
DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

say() { printf '\033[1m==>\033[0m %s\n' "$*"; }

# ── 0. preflight ────────────────────────────────────────────────────────────
say "Preflight"
ssh -o ConnectTimeout=10 "$REMOTE" true || { echo "cannot reach $REMOTE"; exit 1; }

# Bracket the first character of each alternative so the pattern cannot match the
# shell that pgrep itself is running in (the remote command string contains it).
RUNNING=$(ssh "$REMOTE" "pgrep -af '[t]rain_sae|[c]ollect_activations|[r]un_patching|[r]un_ablation|[r]un_circuit|[r]un_path_patching' || true")
if [[ -n "$RUNNING" ]]; then
  echo "REFUSING: a job is already running on $REMOTE:"
  echo "$RUNNING"
  exit 1
fi

FREE=$(ssh "$REMOTE" "df -g /System/Volumes/Data | tail -1 | awk '{print \$4}'")
say "free disk on $REMOTE: ${FREE} GB (need ~12 GB for two models)"
[[ "$FREE" -lt 20 ]] && { echo "insufficient disk"; exit 1; }

# reverse SSH must work, or artifacts cannot come home
if ! ssh "$REMOTE" "ssh -o ConnectTimeout=8 -o BatchMode=yes ${BACK_HOST} true" 2>/dev/null; then
  echo "WARNING: $REMOTE cannot ssh back to ${BACK_HOST}."
  echo "         The wrapper will still run, but will stage outputs in"
  echo "         ${RDIR}/.outbox/${JOB}/ for manual retrieval instead of pushing them."
  PUSH_BACK=0
else
  PUSH_BACK=1
  say "reverse SSH to ${BACK_HOST} OK — outputs will be pushed home automatically"
fi

if [[ $DRY -eq 1 ]]; then say "dry run, stopping here"; exit 0; fi

# ── 1. push code and datasets ───────────────────────────────────────────────
say "Pushing scripts and datasets"
ssh "$REMOTE" "mkdir -p ${RDIR}/data/ioi ${RDIR}/data/factual-assoc ${RDIR}/.outbox/${JOB}"
rsync -az \
  "$LOCAL"/data/ioi/{run_patching_llama3b.py,run_ablation_llama3b.py,run_patching_pythia1b.py,run_ablation_pythia1b.py,run_path_patching_llama3b.py,run_path_patching_pythia1b.py,run_circuit_faithfulness.py,generate_dataset_v2.py,dataset.json} \
  "$REMOTE:${RDIR}/data/ioi/"
say "pushed"

# ── 2. build the wrapper ────────────────────────────────────────────────────
say "Building wrapper for $JOB"
ssh "$REMOTE" "mkdir -p ~/.pacer-done && cat > /tmp/${JOB}.sh" <<WRAPPER
#!/bin/bash
# IOI experiment suite. Stages run sequentially — never two model jobs at once on
# this box (unified-memory bandwidth contention drops throughput ~8x).
set -o pipefail
cd ${RDIR}
LOG=${RDIR}/data/ioi/${JOB}.log
exec >> "\$LOG" 2>&1
echo "=== ${JOB} started \$(date -u +%FT%TZ) ==="

PUSH_BACK=${PUSH_BACK}
BACK_HOST=${BACK_HOST}
OUTBOX=${RDIR}/.outbox/${JOB}

# Copy a produced artifact home immediately, and always stage a copy in the
# outbox so nothing is lost if the reverse link is down.
ship() {
  for f in "\$@"; do
    [ -f "\$f" ] || { echo "  ship: MISSING \$f"; continue; }
    cp "\$f" "\$OUTBOX/" 2>/dev/null
    if [ "\$PUSH_BACK" = "1" ]; then
      rsync -az "\$f" "\${BACK_HOST}:${LOCAL}/\$(dirname \${f#${RDIR}/})/" \\
        && echo "  shipped \$(basename \$f) -> \${BACK_HOST}" \\
        || echo "  SHIP FAILED \$(basename \$f) (staged in outbox)"
    fi
  done
}

stage() {
  local name="\$1"; shift
  echo ""
  echo "--- STAGE \$name  \$(date -u +%FT%TZ) ---"
  "\$@"
  local rc=\$?
  echo "--- STAGE \$name exit=\$rc ---"
  return \$rc
}

FAILED=""

# Stage 1 — Pythia path patching (existing dataset). ~1 h.
stage pythia-path-patching python3 data/ioi/run_path_patching_pythia1b.py \\
  || FAILED="\$FAILED pythia-path-patching"
ship ${RDIR}/data/ioi/path-patching-pythia1b-real.json

# Stage 2 — circuit faithfulness / minimality / completeness, existing dataset.
stage faithfulness-llama-v1 python3 data/ioi/run_circuit_faithfulness.py --model llama \\
  || FAILED="\$FAILED faithfulness-llama-v1"
ship ${RDIR}/data/ioi/faithfulness-llama3b.json

stage faithfulness-pythia-v1 python3 data/ioi/run_circuit_faithfulness.py --model pythia \\
  || FAILED="\$FAILED faithfulness-pythia-v1"
ship ${RDIR}/data/ioi/faithfulness-pythia1b.json

# Stage 3 — multi-template dataset (15 templates, balanced ABBA/BABA).
stage build-dataset-v2 python3 data/ioi/generate_dataset_v2.py --n 200 \\
  || FAILED="\$FAILED build-dataset-v2"
ship ${RDIR}/data/ioi/dataset-v2.json

# Stage 4 — full patching + ablation sweeps on the multi-template set. The long one.
for m in llama3b pythia1b; do
  stage patching-\$m-v2 python3 data/ioi/run_patching_\$m.py --dataset dataset-v2.json \\
    --out patching-\$m-v2.json || FAILED="\$FAILED patching-\$m-v2"
  ship ${RDIR}/data/ioi/patching-\$m-v2.json
  stage ablation-\$m-v2 python3 data/ioi/run_ablation_\$m.py --dataset dataset-v2.json \\
    --out ablation-\$m-v2.json || FAILED="\$FAILED ablation-\$m-v2"
  ship ${RDIR}/data/ioi/ablation-\$m-v2.json
done

# Stage 5 — faithfulness on the multi-template dataset.
stage faithfulness-llama-v2 python3 data/ioi/run_circuit_faithfulness.py --model llama \\
  --dataset dataset-v2.json || FAILED="\$FAILED faithfulness-llama-v2"
mv -f ${RDIR}/data/ioi/faithfulness-llama3b.json ${RDIR}/data/ioi/faithfulness-llama3b-v2.json 2>/dev/null
ship ${RDIR}/data/ioi/faithfulness-llama3b-v2.json

stage faithfulness-pythia-v2 python3 data/ioi/run_circuit_faithfulness.py --model pythia \\
  --dataset dataset-v2.json || FAILED="\$FAILED faithfulness-pythia-v2"
mv -f ${RDIR}/data/ioi/faithfulness-pythia1b.json ${RDIR}/data/ioi/faithfulness-pythia1b-v2.json 2>/dev/null
ship ${RDIR}/data/ioi/faithfulness-pythia1b-v2.json

ship "\$LOG"

# Sentinel LAST, and only after shipping. It records what actually landed.
LANDED=\$(ls -1 "\$OUTBOX" 2>/dev/null | tr '\\n' ' ')
if [ -z "\$FAILED" ]; then
  echo "{\"success\":true,\"job\":\"${JOB}\",\"artifacts\":\"\$LANDED\"}" > ~/.pacer-done/${JOB}.json
else
  echo "{\"success\":false,\"job\":\"${JOB}\",\"failed_stages\":\"\$FAILED\",\"artifacts\":\"\$LANDED\"}" > ~/.pacer-done/${JOB}.json
fi
echo "=== ${JOB} finished \$(date -u +%FT%TZ) failed:'\$FAILED' ==="
WRAPPER

# ── 3. launch detached ──────────────────────────────────────────────────────
say "Launching detached"
PID=$(ssh "$REMOTE" "chmod +x /tmp/${JOB}.sh && nohup /tmp/${JOB}.sh > /tmp/${JOB}.out 2>&1 & echo \$!")
say "pid=$PID"

# ── 4. verify it actually started (up to 12 min for the model download) ─────
say "Waiting for real progress (model download can take several minutes)"
LOGF="${RDIR}/data/ioi/${JOB}.log"
for i in $(seq 1 72); do
  sleep 10
  if ! ssh "$REMOTE" "kill -0 $PID 2>/dev/null"; then
    echo "PROCESS DIED — last 30 lines:"
    ssh "$REMOTE" "tail -30 $LOGF /tmp/${JOB}.out 2>/dev/null"
    exit 1
  fi
  OUT=$(ssh "$REMOTE" "tail -5 $LOGF 2>/dev/null" || true)
  if echo "$OUT" | grep -qiE "traceback|error:|no module|out of memory"; then
    echo "ERROR IN LOG:"; ssh "$REMOTE" "tail -40 $LOGF"; exit 1
  fi
  if echo "$OUT" | grep -qE "STAGE |Loading |edge |Edges to test"; then
    say "progress confirmed"
    ssh "$REMOTE" "tail -6 $LOGF"
    break
  fi
done

FIRST=$(ssh "$REMOTE" "grep -m1 -E 'STAGE |Loading ' $LOGF 2>/dev/null | head -1" || echo "starting")
echo
echo "REMOTE STARTED: ${JOB} pid=${PID} host=${REMOTE} ${FIRST}"
echo
echo "Monitor:  ssh ${REMOTE} 'tail -f ${LOGF}'"
echo "Sentinel: ssh ${REMOTE} 'cat ~/.pacer-done/${JOB}.json'"
echo "Outbox:   ssh ${REMOTE} 'ls -la ${RDIR}/.outbox/${JOB}/'"
