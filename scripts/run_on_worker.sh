#!/bin/bash
# VM-side: run a repo script on this worker with the shared env, logging to NFS.
# Also runs the TPU hardware monitor sidecar (HBM, duty cycle) for the lifetime of the run.
# usage: run_on_worker.sh <run_name> <repo-relative script> [VAR=value ...] [args...]
#   leading UPPER_CASE=value args are exported as env vars (e.g. MAX_STEPS=4 SAVE_EVERY=2)
name=$1; script=$2; shift 2
while [[ "${1:-}" =~ ^[A-Z_][A-Z0-9_]*= ]]; do export "$1"; shift; done
source "$(dirname "$0")/vm_env.sh"
worker=$(hostname | grep -o 'w-[0-9]*$')
mkdir -p "$LOG_ROOT/$name"
export RUN_NAME=$name WORKER=$worker

"$NFS_ROOT/venvs/tools/bin/python" "$REPO/tools/hw_monitor.py" "$LOG_ROOT/$name/hw-$worker.csv" \
  > /dev/null 2>&1 &
monitor=$!
trap 'kill $monitor 2>/dev/null' EXIT

# stdin from /dev/null: any interactive prompt (logins etc.) fails immediately instead of hanging the run
bash "$REPO/$script" "$@" < /dev/null > "$LOG_ROOT/$name/$worker.log" 2>&1
