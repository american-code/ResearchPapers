#!/bin/bash
# Pull IOI experiment artifacts down from lab-02 as they are produced.
#
# lab-02 cannot SSH back to this machine (Remote Login is off here), so the remote
# wrapper stages every output in ~/ResearchPapers/.outbox/<job>/ instead of pushing.
# This poller closes the loop from this side. Run it alongside the remote job, or
# any time afterwards to collect whatever has landed.
#
# This exists because the Jul-29 SAE runs completed on lab-02 and were never
# retrieved: the pipeline had a completion signal but no artifact retrieval.
#
# Usage:
#   ./scripts/pull_ioi_results.sh                 # newest job, poll until done
#   ./scripts/pull_ioi_results.sh <job-id>        # a specific job
#   ./scripts/pull_ioi_results.sh <job-id> once   # single pull, no polling

set -uo pipefail

REMOTE=lab-02
RUSER=lab-02
RDIR="/Users/${RUSER}/ResearchPapers"
LOCAL="/Users/melton/ResearchPapers"
INTERVAL="${INTERVAL:-300}"

JOB="${1:-}"
MODE="${2:-poll}"

if [[ -z "$JOB" ]]; then
  JOB=$(ssh "$REMOTE" "ls -1t ${RDIR}/.outbox 2>/dev/null | head -1")
  [[ -z "$JOB" ]] && { echo "no jobs found in ${RDIR}/.outbox"; exit 1; }
  echo "using newest job: $JOB"
fi

OUTBOX="${RDIR}/.outbox/${JOB}"
SENTINEL="~/.pacer-done/${JOB}.json"

pull() {
  # Route each artifact to the directory it belongs in.
  local n=0
  local files
  files=$(ssh "$REMOTE" "ls -1 ${OUTBOX} 2>/dev/null" || true)
  [[ -z "$files" ]] && return 0
  while IFS= read -r f; do
    [[ -z "$f" ]] && continue
    case "$f" in
      *.log)                    dest="$LOCAL/data/ioi" ;;
      dataset-v2.json)          dest="$LOCAL/data/ioi" ;;
      faithfulness-*|patching-*|ablation-*|path-patching-*) dest="$LOCAL/data/ioi" ;;
      *)                        dest="$LOCAL/data/ioi" ;;
    esac
    mkdir -p "$dest"
    if rsync -az --checksum "$REMOTE:${OUTBOX}/${f}" "$dest/" 2>/dev/null; then
      n=$((n+1))
    fi
  done <<< "$files"
  echo "  pulled/verified $n artifact(s) -> data/ioi/"
}

echo "watching $JOB on $REMOTE (interval ${INTERVAL}s)"
while true; do
  ts=$(date -u +%FT%TZ)
  stage=$(ssh "$REMOTE" "grep -E '^--- STAGE' ${RDIR}/data/ioi/${JOB}.log 2>/dev/null | tail -1" || true)
  echo "[$ts] ${stage:-no stage line yet}"
  pull

  if ssh "$REMOTE" "test -f ${SENTINEL}" 2>/dev/null; then
    echo
    echo "=== SENTINEL ==="
    ssh "$REMOTE" "cat ${SENTINEL}"
    echo
    pull
    echo "job complete; artifacts in $LOCAL/data/ioi/"
    exit 0
  fi

  if ! ssh "$REMOTE" "pgrep -f '[i]oi-suite' >/dev/null" 2>/dev/null; then
    echo "WARNING: no ioi-suite process running and no sentinel written."
    echo "         Job may have died. Last 25 log lines:"
    ssh "$REMOTE" "tail -25 ${RDIR}/data/ioi/${JOB}.log 2>/dev/null"
    exit 1
  fi

  [[ "$MODE" == "once" ]] && exit 0
  sleep "$INTERVAL"
done
