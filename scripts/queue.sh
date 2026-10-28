#!/bin/bash
# VM-side job queue for a multi-host slice: a night of runs needs no laptop connection.
# Start once on all workers:  scripts/tpu.sh all "tmux new -d -s q-<name> 'bash \$NFS_ROOT/llm-gcrl/scripts/queue.sh <name>'"
# Queue file $NFS_ROOT/queues/<name>.txt, re-read before every job, one job per line:
#   <run_name> <cap_minutes> <repo-relative script> [VAR=value ...] [args...]
# '#' starts a comment; a line STOP ends the queue; at the end of the file the queue waits for more lines.
# Per job: worker 0 git-pulls the NFS checkout and announces the job; every worker runs it via run_on_worker.sh.
# When any worker's job exits, the cap passes, or $STATE/<seq>.abort appears, the remaining workers are killed
# after GRACE_S (one crashed process leaves the others hanging in collectives) and their TPU holders cleaned up.
# Worker 0 then appends tools/run_summary.py output to $LOG_ROOT/queue-<name>/summary.txt.
set -uo pipefail
qname=$1
source "$(dirname "$0")/vm_env.sh"
QFILE=$NFS_ROOT/queues/$qname.txt
STATE=$NFS_ROOT/queues/$qname.state
OUT=$LOG_ROOT/queue-$qname
NW=${QUEUE_WORKERS:-8}
GRACE_S=${GRACE_S:-300}
W=$(curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number)
mkdir -p "$STATE" "$OUT"
exec >> "$OUT/w$W.log" 2>&1
touch "$STATE/completed"
log() { echo "$(date -u +%m-%d_%H:%M:%S) w$W $*"; }

next_line() {  # first job line whose run name is not completed (empty: none yet; STOP: end)
  [ -f "$QFILE" ] || return 0
  while read -r line; do
    line=${line%%#*}; read -r name _ <<< "$line"
    [ -z "$name" ] && continue
    [ "$name" = STOP ] && { echo STOP; return; }
    grep -qxF "$name" "$STATE/completed" || { echo "$line"; return; }
  done < "$QFILE"
}

free_tpu() {
  local pids
  pids=$(sudo lsof -t /dev/vfio/* 2>/dev/null | sort -u)
  [ -n "$pids" ] && { log "killing TPU holders: $pids"; sudo kill -9 $pids 2>/dev/null; sleep 5; }
}

last_seq=$(cat "$STATE/current" 2>/dev/null | cut -d'|' -f1)
[ "$W" = 0 ] && last_seq=""
while true; do
  if [ "$W" = 0 ]; then
    git -C "$REPO" pull --ff-only -q 2>&1 | tail -2
    line=$(next_line)
    if [ -z "$line" ]; then sleep 120; continue; fi
    seq=$(date +%s)
    echo "$seq|$line" > "$STATE/current.tmp" && mv "$STATE/current.tmp" "$STATE/current"
  else
    cur=$(cat "$STATE/current" 2>/dev/null)
    seq=${cur%%|*}; line=${cur#*|}
    if [ -z "$cur" ] || [ "$seq" = "$last_seq" ]; then sleep 10; continue; fi
  fi
  [ "$line" = STOP ] && { log "STOP"; exit 0; }
  read -r name cap script args <<< "$line"
  log "start $name (cap ${cap}m): $script $args"
  t0=$(date +%s)
  free_tpu
  # word-splitting of $args is intended (VAR=value tokens and Tunix overrides without spaces)
  setsid bash "$REPO/scripts/run_on_worker.sh" "$name" "$script" $args &
  pid=$!
  end_seen=""
  while kill -0 "$pid" 2>/dev/null; do
    sleep 15
    now=$(date +%s)
    if [ -z "$end_seen" ] && { ls "$STATE/$seq".done.* > /dev/null 2>&1 || [ -f "$STATE/$seq.abort" ] \
        || [ $((now - t0)) -gt $((cap * 60)) ]; }; then
      end_seen=$now; log "another worker ended / abort / cap: grace ${GRACE_S}s"
    fi
    if [ -n "$end_seen" ] && [ $((now - end_seen)) -gt "$GRACE_S" ]; then
      log "killing $name"; kill -TERM -- -"$pid" 2>/dev/null; sleep 15; kill -9 -- -"$pid" 2>/dev/null; break
    fi
  done
  wait "$pid" 2>/dev/null; rc=$?
  touch "$STATE/$seq.done.w$W"
  sleep 20; free_tpu
  touch "$STATE/$seq.clean.w$W"
  log "end $name rc=$rc after $(( ($(date +%s) - t0) / 60 ))m"
  last_seq=$seq
  if [ "$W" = 0 ]; then
    for _ in $(seq 90); do  # all workers clean (<= 15 min)
      [ "$(ls "$STATE/$seq".clean.* 2>/dev/null | wc -l)" -ge "$NW" ] && break; sleep 10
    done
    { echo "=== $name  $(date -u +%m-%d_%H:%M)  $(( ($(date +%s) - t0) / 60 ))m  rc=$rc  :: $script $args"
      "$NFS_ROOT/venvs/tools/bin/python" "$REPO/tools/run_summary.py" "$LOG_ROOT/$name" 2>&1 | head -40
    } >> "$OUT/summary.txt"
    echo "$name" >> "$STATE/completed"
  else
    # wait for worker 0 to announce the next job (or finish the summary)
    for _ in $(seq 120); do
      cur=$(cat "$STATE/current" 2>/dev/null); [ "${cur%%|*}" != "$seq" ] && break; sleep 10
    done
  fi
done
