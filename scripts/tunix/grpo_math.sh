#!/bin/bash
# Tunix GRPO + vLLM (tpu-inference) rollouts on POLARIS math. Standard profiling workload: 16 prompts x 8 rollouts.
# Launch on ALL workers from the laptop:
#   scripts/tpu.sh launch tunix-grpo scripts/tunix/grpo_math.sh [VAR=value ...] [tunix key=value overrides...]
# Settings (any of them can be given per worker as VAR_W<n>, e.g. MODE_W1=full, so the 4 hosts of a
# single-host run benchmark 4 configurations at once):
#   HOST_MODE=single|multi   single: each host is an independent 4-chip TPU (default); multi: one 16-chip job
#   MODE=lora|full           LoRA (r=32) or full fine-tuning (default lora)
#   MODEL=Qwen/Qwen3-4B      HF id (Qwen3 dense or MoE)
#   MAX_TOKENS=1024 MAX_PROMPT=768 NUM_BATCHES=4 VLLM_HBM=0.4 MESH=1x4 VLLM_MAX_SEQS=128 VLLM_MAX_BATCHED=8192
#   TRAIN_MICRO=1 LOGPS_MICRO=1 ROLLOUT_MICRO=16   (micro-batches, in prompts)   OFFLOAD=false FORCE_FULL=0
# Storage: HF weights -> worker-local $SCRATCH; checkpoints/tensorboard -> $CKPT_ROOT/tunix/$RUN_NAME[/w-N] (GCS);
# metrics -> wandb + $LOG_ROOT/$RUN_NAME/eff[-w-N].jsonl (NFS).
set -euo pipefail

WORKER_ID=$(curl -s -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/agent-worker-number)
pick() {  # per-worker override VAR_W<id>, else VAR, else default
  local v="${1}_W${WORKER_ID}"
  if [ -n "${!v:-}" ]; then echo "${!v}"; elif [ -n "${!1:-}" ]; then echo "${!1}"; else echo "$2"; fi
}
HOST_MODE=$(pick HOST_MODE single)
MODE=$(pick MODE lora)
ALGO=$(pick ALGO grpo)               # grpo | ppo (actor + LoRA critic, configs/tunix/ppo_qwen3_polaris.yaml)
MODEL=$(pick MODEL Qwen/Qwen3-4B)
export MODEL   # effmetrics reads it for the MFU parameter count
MAX_TOKENS=$(pick MAX_TOKENS 1024)
MAX_PROMPT=$(pick MAX_PROMPT 768)
NUM_BATCHES=$(pick NUM_BATCHES 4)
VLLM_HBM=$(pick VLLM_HBM 0.4)
MESH=$(pick MESH 1x4)                  # fsdp x tp, written AxB so it survives ssh/tmux quoting
MESH_TUPLE="(${MESH/x/,})"
BATCH=$(pick BATCH 16)                 # prompts per RL step
GROUP=$(pick GROUP 8)                  # rollouts per prompt (GRPO group size)
VLLM_MAX_SEQS=$(pick VLLM_MAX_SEQS $((BATCH * GROUP)))   # concurrent sequences (= rollouts per step)
VLLM_MAX_BATCHED=$(pick VLLM_MAX_BATCHED 8192)   # tokens per vLLM forward (bounds prefill temporaries)
TRAIN_MICRO=$(pick TRAIN_MICRO 1)    # prompts (x num_generations sequences) per gradient micro-step
LOGPS_MICRO=$(pick LOGPS_MICRO 1)    # prompts per reference/old-logprob micro-pass
ROLLOUT_MICRO=$(pick ROLLOUT_MICRO $BATCH)  # prompts handed to vLLM at once (16 -> all 128 rollouts concurrently)
OFFLOAD=$(pick OFFLOAD false)         # Tunix offload_to_cpu: park inactive models in host RAM between phases
FORCE_FULL=$(pick FORCE_FULL 0)       # 1: every rollout generates exactly MAX_TOKENS (worst case for max-context tests)
VLLM_RESHARD_CHUNK=$(pick VLLM_RESHARD_CHUNK 16)  # weight sync in chunks of N tensors, deleting old vLLM buffers per chunk
LOGPS_CHUNK=$(pick LOGPS_CHUNK 0)    # >0: compute logits/logprobs in sequence chunks of N tokens (rematerialized; saves HBM)
MODEL_NAME=$(echo "${MODEL#*/}" | tr 'A-Z' 'a-z')   # Qwen/Qwen3-4B -> qwen3-4b (Tunix model name)
# Checkpoints only matter for real training: profiling runs of models above 4B skip them (bucket cost).
case "$MODEL" in *-0.6B|*-1.7B|*-4B) SAVE_CKPT=${SAVE_CKPT:-1} ;; *) SAVE_CKPT=${SAVE_CKPT:-0} ;; esac
export SAVE_CKPT

if [ "$HOST_MODE" = single ]; then
  source "$REPO/scripts/single_host_env.sh"
  OUT=$CKPT_ROOT/tunix/$RUN_NAME/$WORKER
  WANDB_RUN=$RUN_NAME-$WORKER
  export VLLM_XLA_CACHE_PATH=$SCRATCH/vllm_xla_cache_single   # vLLM's own XLA cache, kept per topology
else
  OUT=$CKPT_ROOT/tunix/$RUN_NAME
  WANDB_RUN=$RUN_NAME
  # Per-host vLLM programs inside a multi-host job crash libtpu (tpu_topology.cc ValidateIndexForBounds) when
  # read back from a persistent cache, so compile them fresh every run.
  export VLLM_DISABLE_COMPILE_CACHE=1 JAX_ENABLE_COMPILATION_CACHE=false
fi
echo "worker=$WORKER algo=$ALGO host_mode=$HOST_MODE mode=$MODE model=$MODEL mesh=$MESH max_tokens=$MAX_TOKENS vllm_hbm=$VLLM_HBM logps_chunk=$LOGPS_CHUNK offload=$OFFLOAD"
# Fresh metrics for this launch (a crashed earlier attempt under the same run name must not leak in).
if [ "$HOST_MODE" = single ]; then rm -f "$LOG_ROOT/$RUN_NAME/eff-$WORKER.jsonl"
elif [ "$WORKER_ID" = 0 ]; then rm -f "$LOG_ROOT/$RUN_NAME/eff.jsonl"; fi

LORA_ARGS=()
FORCE_ARGS=()
if [ "$FORCE_FULL" = 1 ]; then
  FORCE_ARGS=("rollout_config.rollout_vllm_sampling_kwargs={'min_tokens': $MAX_TOKENS}")
fi
if [ "$MODE" = lora ]; then
  export LORA_RANK=32 LORA_ALPHA=64.0   # tools/tunix_run.py merges LoRA into the weights synced to vLLM with alpha/rank
  LORA_ARGS=("actor_model_config.lora_config={'rank': $LORA_RANK, 'alpha': $LORA_ALPHA, 'module_path': '.*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj'}")
fi

source "$NFS_ROOT/venvs/maxtext/bin/activate"
TUNIX_SRC=$NFS_ROOT/third_party/tunix      # same commit as the installed package; only used for base_config.yaml
if [ -z "${WANDB_API_KEY:-}" ]; then export WANDB_MODE=offline; fi
export HF_TOKEN=${HF_TOKEN:-}               # Tunix's HF download prompts for login unless HF_TOKEN is set
cd "$SCRATCH"                               # keep cwd off the tunix source tree

# Empty root directory = Tunix creates no checkpointer at all.
if [ "$SAVE_CKPT" = 1 ]; then CKPT_DIR=$OUT/checkpoints; else CKPT_DIR=""; fi

# One copy per model in the worker-local HF cache, shared with SkyRL (no second local_dir copy).
MODEL_DIR=$(python3 -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))")

if [ "$ALGO" = ppo ]; then
  export PPO_GROUP=$GROUP
  ALGO_ARGS=()
else
  ALGO_ARGS=("grpo_config.num_generations=$GROUP")
fi

TUNIX_ALGO=$ALGO python3 "$REPO/tools/tunix_run.py" \
  "$TUNIX_SRC/tunix/cli/base_config.yaml" \
  override_config_file="$REPO/configs/tunix/${ALGO}_qwen3_polaris.yaml" \
  "reward_functions=[$REPO/tools/math_reward.py]" \
  model_config.model_name="$MODEL_NAME" \
  model_config.model_id="$MODEL" \
  model_config.model_download_path="$MODEL_DIR" \
  model_config.intermediate_ckpt_dir="$MODEL_ROOT/tunix-intermediate/$MODEL_NAME" \
  tokenizer_config.tokenizer_path="$MODEL_DIR" \
  actor_model_config.mesh.shape="$MESH_TUPLE" \
  num_batches="$NUM_BATCHES" \
  rl_training_config.max_steps="$NUM_BATCHES" \
  rl_training_config.checkpoint_root_directory="$CKPT_DIR" \
  rl_training_config.metrics_logging_options.log_dir="$OUT/tensorboard" \
  rl_training_config.metrics_logging_options.run_name="$WANDB_RUN" \
  rollout_config.total_generation_steps="$MAX_TOKENS" \
  rollout_config.max_prompt_length="$MAX_PROMPT" \
  vllm_config.hbm_utilization="$VLLM_HBM" \
  vllm_config.max_num_seqs="$VLLM_MAX_SEQS" \
  vllm_config.max_num_batched_tokens="$VLLM_MAX_BATCHED" \
  batch_size="$BATCH" \
  rl_training_config.mini_batch_size="$BATCH" \
  "${ALGO_ARGS[@]}" \
  rl_training_config.train_micro_batch_size="$TRAIN_MICRO" \
  rl_training_config.compute_logps_micro_batch_size="$LOGPS_MICRO" \
  rl_training_config.rollout_micro_batch_size="$ROLLOUT_MICRO" \
  offload_to_cpu="$OFFLOAD" \
  rollout_config.rollout_vllm_delete_dst_buffers=true \
  rollout_config.rollout_vllm_reshard_chunk_size="$VLLM_RESHARD_CHUNK" \
  rl_training_config.compute_logps_chunk_size="$LOGPS_CHUNK" \
  "${LORA_ARGS[@]}" \
  "${FORCE_ARGS[@]}" \
  "$@"
