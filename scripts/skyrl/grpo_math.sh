#!/bin/bash
# SkyRL (JAX backend, Tinker API) GRPO smoke test: Qwen3-4B + LoRA on math RL (default POLARIS-53K),
# multi-host v5e-16. Launch on ALL workers from the laptop:
#   scripts/tpu.sh launch skyrl-grpo scripts/skyrl/grpo_math.sh [DATASET=polaris MAX_TOKENS=1024 MAX_STEPS=10 SAVE_EVERY=5] [cookbook overrides...]
#   KV probe instead of training: ... grpo_math.sh CLIENT=kvprobe [KV_LENGTHS=512,1024,2048]
#   Max-context probe: ... grpo_math.sh CLIENT=maxctx MAXCTX_MODE=sample|train [NUM_SAMPLES=128 CTX_LENGTHS=1920,2944,...]
#
# Worker 0 runs the Tinker API server (JAX process 0) plus the RL client loop; workers 1-3 run
# JAX backend workers. Storage:
#   - LoRA sampler weights + training checkpoints -> $CKPT_ROOT/skyrl/$RUN_NAME (GCS, same region)
#   - HF base weights / dataset cache             -> $HF_HOME (worker-local disk)
#   - client metrics.jsonl + logs                 -> $LOG_ROOT/$RUN_NAME (NFS, small text)
set -euo pipefail

SKYRL_DIR=$NFS_ROOT/third_party/SkyRL
PY=$SKYRL_DIR/.venv/bin/python
MODEL=${MODEL:-Qwen/Qwen3-4B}
export MODEL   # effmetrics reads it for the MFU parameter count
# Checkpoints only matter for real training: profiling runs of models above 4B skip them (bucket cost).
case "$MODEL" in *-0.6B|*-1.7B|*-4B) SAVE_CKPT=${SAVE_CKPT:-1} ;; *) SAVE_CKPT=${SAVE_CKPT:-0} ;; esac
export SAVE_CKPT
LORA_RANK=${LORA_RANK:-32}
DATASET=${DATASET:-polaris}       # cookbook math_rl env: polaris | deepmath | math | gsm8k
MAX_TOKENS=${MAX_TOKENS:-1024}
# Concurrency knobs (smaller for big / MoE models): sequences sampled at once, sequences per training micro-batch
SAMPLE_MAX_SEQS=${SAMPLE_MAX_SEQS:-256}
TRAIN_MICRO=${TRAIN_MICRO:-8}
# Mesh (FSDP x EP x TP = 16 chips); MoE models want expert parallelism (EP) instead of FSDP
TP=${TP:-4} FSDP=${FSDP:-4} EP=${EP:-1}

md() { curl -s -H "Metadata-Flavor: Google" "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1"; }
WORKER_ID=$(md agent-worker-number)
NUM_WORKERS=$(md worker-network-endpoints | awk -F, '{print NF}')
COORD_IP=$(md worker-network-endpoints | cut -d, -f1 | awk -F: '{print $NF}')
COORD=$COORD_IP:7777
echo "worker $WORKER_ID/$NUM_WORKERS coordinator $COORD run $RUN_NAME"

# SkyRL stages every sampler-weight save (GBs for MoE) in a temp dir that a crashed engine never removes:
# keep temp files per run on scratch and drop earlier runs' leftovers (one run at a time on the slice).
rm -rf "$SCRATCH/tmp"; mkdir -p "$SCRATCH/tmp/$RUN_NAME"; export TMPDIR=$SCRATCH/tmp/$RUN_NAME

DONE=$LOG_ROOT/$RUN_NAME/DONE   # written by worker 0 on exit; backend workers never exit on their own
rm -f "$DONE"

if [ "$WORKER_ID" != "0" ]; then
  # Give the coordinator a head start, then join as a backend worker until worker 0 is done.
  sleep 30
  "$PY" -m skyrl.backends.jax --coordinator-address "$COORD" --num-processes "$NUM_WORKERS" --process-id "$WORKER_ID" &
  WORKER_PID=$!
  while kill -0 $WORKER_PID 2>/dev/null && [ ! -f "$DONE" ]; do sleep 10; done
  kill $WORKER_PID 2>/dev/null; sleep 5; kill -9 $WORKER_PID 2>/dev/null
  exit 0
fi

# ---- worker 0: API server (JAX process 0) ----
BACKEND_CONFIG=$(cat <<EOF
{"tensor_parallel_size": ${TP:-4}, "fully_sharded_data_parallel_size": ${FSDP:-4}, "expert_parallel_size": ${EP:-1},
 "num_processes": $NUM_WORKERS, "coordinator_address": "$COORD",
 "max_lora_adapters": 2, "max_lora_rank": $LORA_RANK,
 "train_micro_batch_size": ${TRAIN_MICRO:-8}, "sample_max_num_sequences": ${SAMPLE_MAX_SEQS:-256},
 "gradient_checkpointing": true}
EOF
)
export SKYRL_DATABASE_URL=sqlite:///$SCRATCH/skyrl-$RUN_NAME.db   # server state, worker-local
rm -f "$SCRATCH/skyrl-$RUN_NAME.db"
# The API server re-launches its engine as `uv run <same uv flags> -m skyrl.tinker.engine`, so it must be
# started through uv. --no-sync keeps uv from re-syncing the venv (which would drop the cookbook client).
export PATH=$CONDA_ROOT/envs/llm-gcrl/bin:$PATH UV_CACHE_DIR=$SCRATCH/uv-cache UV_PYTHON_PREFERENCE=only-system
cd "$SKYRL_DIR"
# Bind to localhost: the project firewall exposes every VM port to the internet.
uv run --no-sync --extra tpu --extra tinker --extra jax --extra gcp -m skyrl.tinker.api \
  --host 127.0.0.1 \
  --base-model "$MODEL" \
  --backend jax \
  --backend-config "$BACKEND_CONFIG" \
  --checkpoints-base "$CKPT_ROOT/skyrl/$RUN_NAME" \
  --session-timeout-sec 86400 \
  > "$LOG_ROOT/$RUN_NAME/server.log" 2>&1 &
SERVER_PID=$!
cleanup() {
  touch "$DONE"
  kill $SERVER_PID 2>/dev/null
  # the engine is a separate `uv run` child that can outlive the API server
  pkill -f "[s]kyrl.tinker.engine.*$RUN_NAME" 2>/dev/null; sleep 5
  pkill -9 -f "[s]kyrl.tinker.engine.*$RUN_NAME" 2>/dev/null; true
}
trap cleanup EXIT

echo "waiting for API server (model download + load on all hosts)..."
until curl -s -o /dev/null 127.0.0.1:8000; do
  kill -0 $SERVER_PID 2>/dev/null || { echo "server died; see server.log"; tail -n 50 "$LOG_ROOT/$RUN_NAME/server.log"; exit 1; }
  sleep 15
done
echo "server up"
# If the server/engine dies (e.g. OOM), the cookbook client retries forever instead of exiting, holding the
# slice: stop the client so the run ends and cleanup runs.
( while kill -0 $SERVER_PID 2>/dev/null; do sleep 15; done
  echo "API server exited; stopping client (see server.log)"
  pkill -P $$ -f 'skyrl_(client|kv_probe|maxctx_probe)' ) &

# ---- worker 0: RL client (tinker-cookbook math_rl) ----
# GRPO = group-centred advantages (cookbook default) + PPO-clip loss (loss_fn=ppo), no critic.
# Thinking is disabled so answers fit in max_tokens.
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "WANDB_API_KEY not set: logging wandb offline to $WANDB_DIR (sync later with 'wandb sync')"
  export WANDB_MODE=offline
fi
export TINKER_API_KEY=tml-dummy
if [ "${CLIENT:-train}" = "kvprobe" ]; then
  # Sampling-only KV-cache / decode-throughput probe instead of training (tools/kv_report.py analyses it)
  "$PY" "$REPO/tools/skyrl_kv_probe.py" --base-url http://127.0.0.1:8000 --model "$MODEL" \
    --lengths $(echo "${KV_LENGTHS:-512,1024,2048}" | tr , " ")
  echo "CLIENT_DONE"
  exit 0
fi
if [ "${CLIENT:-train}" = "maxctx" ]; then
  # Max-context probe, increasing L until OOM: MAXCTX_MODE=sample (all NUM_SAMPLES rollouts at once) or
  # train (synthetic sequences). Separate launches, since an OOM can take the engine down.
  LEN_ARGS=()
  if [ -n "${CTX_LENGTHS:-}" ]; then LEN_ARGS=(--lengths $(echo "$CTX_LENGTHS" | tr , " ")); fi
  "$PY" "$REPO/tools/skyrl_maxctx_probe.py" --base-url http://127.0.0.1:8000 --model "$MODEL" \
    --mode "${MAXCTX_MODE:-sample}" --num-samples "${NUM_SAMPLES:-128}" "${LEN_ARGS[@]}" || true
  echo "CLIENT_DONE"
  exit 0
fi
"$PY" "$REPO/tools/skyrl_client.py" \
  base_url=http://127.0.0.1:8000 \
  model_name="$MODEL" \
  env=$DATASET \
  renderer_name=qwen3_disable_thinking \
  lora_rank=$LORA_RANK \
  learning_rate=4e-5 \
  loss_fn=ppo \
  group_size=8 \
  groups_per_batch=32 \
  max_tokens=$MAX_TOKENS \
  max_steps=${MAX_STEPS:-10} \
  eval_every=0 \
  save_every=$([ "$SAVE_CKPT" = 1 ] && echo "${SAVE_EVERY:-5}" || echo 0) \
  log_path="$LOG_ROOT/$RUN_NAME/client" \
  wandb_project="$WANDB_PROJECT" \
  wandb_name="$RUN_NAME" \
  behavior_if_log_dir_exists=delete \
  "$@"
echo "CLIENT_DONE"
