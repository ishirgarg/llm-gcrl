#!/bin/bash
# Laptop-side helper for driving the multi-host TPU slice over SSH.
# It only sends commands; it never copies data from the VM or bucket to this machine.
#
#   scripts/tpu.sh all  '<cmd>'        run a command on every worker
#   scripts/tpu.sh w0   '<cmd>'        run a command on worker 0
#   scripts/tpu.sh pull                git pull the NFS checkout (visible to all workers); the only way code reaches
#                                      the VMs: edit + commit + push locally, then pull (keeps SSH traffic minimal)
#   scripts/tpu.sh launch <name> <script> [args...]
#                                      run <script> (repo-relative) on all workers in tmux session <name>
#   scripts/tpu.sh logs <name> [N] [W] tail worker-W log of a launch (last N lines; defaults 50, 0)
#   scripts/tpu.sh stop <name>         kill the run on all workers
#   scripts/tpu.sh tpu-busy            show processes holding the TPU on each worker
set -euo pipefail

# TPU, ZONE, NUM_WORKERS and NFS_ROOT come from the git-ignored site.env at the repo root (see site.env.example).
# A TPU=... / NUM_WORKERS=... already in the environment wins over site.env (drive several slices at once).
_tpu_override=${TPU:-}; _nw_override=${NUM_WORKERS:-}
source "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)/site.env"
TPU=${_tpu_override:-${TPU:?set TPU in site.env}}
NUM_WORKERS=${_nw_override:-${NUM_WORKERS:-8}}
ZONE=${ZONE:?set ZONE in site.env}
REPO=$NFS_ROOT/llm-gcrl
LOG_ROOT=$NFS_ROOT/logs

ssh_workers() {
  local worker=$1; shift
  gcloud alpha compute tpus tpu-vm ssh "$TPU" --zone="$ZONE" --worker="$worker" --command="$*" 2>&1 \
    | { grep -v -e "Warning: Permanently added" -e "Using ssh batch size" -e "SSH: Attempting to connect" || true; }
}

cmd=${1:-}; shift || true
case "$cmd" in
  all) ssh_workers all "$*" ;;
  w0)  ssh_workers 0 "$*" ;;
  pull) ssh_workers 0 "cd $REPO && git pull --ff-only" ;;
  launch)
    name=$1; script=$2; shift 2
    # Each worker writes its own log on NFS so any worker's output can be inspected.
    ssh_workers all "tmux kill-session -t $name 2>/dev/null; \
      tmux new -d -s $name 'bash $REPO/scripts/run_on_worker.sh $name $script $*'"
    # A shell-quoting problem in the args makes tmux fail silently, so verify the session exists everywhere.
    started=$(ssh_workers all "tmux has-session -t $name 2>/dev/null && echo STARTED" | grep -c STARTED || true)
    if [ "$started" -ne "$NUM_WORKERS" ]; then
      echo "ERROR: '$name' started on $started/$NUM_WORKERS workers (check args for quotes/parentheses)" >&2
      exit 1
    fi
    echo "Launched '$name' on all workers. Logs: $LOG_ROOT/$name/w-*.log (scripts/tpu.sh logs $name)"
    ;;
  logs)
    name=$1; n=${2:-50}; w=${3:-0}
    ssh_workers 0 "tail -n $n $LOG_ROOT/$name/w-$w.log"
    ;;
  stop)
    name=$1
    # Every launched process has the run name in its command line (log/checkpoint paths, run_name=...).
    # The remote command must never contain the name as one contiguous string, or pkill kills its own
    # shell: hence the [x]yz pattern and the split "x""yz" session name. Workers are contacted one at a
    # time with a timeout because --worker=all SSH is flaky; leftovers are force-killed.
    pat="[${name:0:1}]${name:1}"
    split="\"${name:0:1}\"\"${name:1}\""
    for w in $(seq 0 $(($NUM_WORKERS - 1))); do
      timeout 90 gcloud alpha compute tpus tpu-vm ssh "$TPU" --zone="$ZONE" --worker="$w" --command="\
        tmux kill-session -t $split 2>/dev/null; sleep 2; pkill -f '$pat'; sleep 5; pkill -9 -f '$pat'; \
        echo \"\$(hostname): stopped, TPU holders left: \$(sudo lsof -w /dev/vfio/* 2>/dev/null | awk 'NR>1' | wc -l)\"" 2>&1 \
        | grep "stopped" || echo "worker $w: ssh failed, retry 'scripts/tpu.sh stop $name'"
    done
    ;;
  tpu-busy) ssh_workers all 'echo "== $(hostname)"; sudo lsof -w /dev/vfio/* 2>/dev/null | awk "NR>1{print \$1, \$2, \$3}" | sort -u || true' ;;
  *) sed -n '2,13p' "$0"; exit 1 ;;
esac
