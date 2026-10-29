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
#   TRAIN_MICRO=auto LOGPS_MICRO=1 ROLLOUT_MICRO=16   (micro-batches, in prompts)   OFFLOAD=false FORCE_FULL=0
#   LOGPS_CHUNK=auto REMAT=3 PREFIX_CACHE=true XLA_OPT=1 BETA= LR= CRITIC_LR= WARMUP=1 DATA=polaris|math
#   EVAL_SETS=math500:4:3072,... EVAL_EVERY=25 (tools/rl_eval.py) VLLM_MAX_LEN=0 TIS=0 TIS_CAP=2.0
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
# LEN_SCHEDULE="4096:2,8192:2,...": profile several completion caps in one job (tools/tunix_run.py); sets
# MAX_TOKENS to the largest cap and NUM_BATCHES to the total number of steps.
LEN_SCHEDULE=$(pick LEN_SCHEDULE "")
export LEN_SCHEDULE
if [ -n "$LEN_SCHEDULE" ]; then
  MAX_TOKENS=0; NUM_BATCHES=0
  for spec in ${LEN_SCHEDULE//,/ }; do
    cap=${spec%%:*}; n=${spec##*:}
    [ "$cap" -gt "$MAX_TOKENS" ] && MAX_TOKENS=$cap
    NUM_BATCHES=$((NUM_BATCHES + n))
  done
  export MAX_TOKENS NUM_BATCHES
fi
MAX_PROMPT=$(pick MAX_PROMPT 768)
NUM_BATCHES=$(pick NUM_BATCHES 4)   # (LEN_SCHEDULE overrides it above via the exported value)
VLLM_HBM=$(pick VLLM_HBM 0.4)
MESH=$(pick MESH 1x4)                  # fsdp x tp, written AxB so it survives ssh/tmux quoting
MESH_TUPLE="(${MESH/x/,})"
BATCH=$(pick BATCH 16)                 # prompts per RL step
GROUP=$(pick GROUP 8)                  # rollouts per prompt (GRPO group size)
VLLM_MAX_SEQS=$(pick VLLM_MAX_SEQS $((BATCH * GROUP)))   # concurrent sequences (= rollouts per step)
VLLM_MAX_BATCHED=$(pick VLLM_MAX_BATCHED 8192)   # tokens per vLLM forward (bounds prefill temporaries)
# prompts (x GROUP sequences) per gradient micro-step. Default: as many as fit a 64k-token budget per micro-step,
# a divisor of BATCH. Bigger micro-steps amortise the per-micro-step FSDP weight all-gathers: at 1k, 1 -> 4 prompts
# makes training 2.7x (GRPO) to 3.3x (PPO) faster.
SEQ_LEN=$(( $(pick MAX_PROMPT 768) + $(pick MAX_TOKENS 1024) ))
MICRO_BUDGET=$(pick MICRO_BUDGET 65536)
auto_micro=1
for m in $(seq "$(pick BATCH 16)" -1 1); do
  if [ $(( $(pick BATCH 16) % m )) -eq 0 ] && [ $(( m * $(pick GROUP 8) * SEQ_LEN )) -le "$MICRO_BUDGET" ]; then auto_micro=$m; break; fi
done
TRAIN_MICRO=$(pick TRAIN_MICRO $auto_micro)
LOGPS_MICRO=$(pick LOGPS_MICRO 1)    # prompts per reference/old-logprob micro-pass
ROLLOUT_MICRO=$(pick ROLLOUT_MICRO $BATCH)  # prompts handed to vLLM at once (16 -> all 128 rollouts concurrently)
OFFLOAD=$(pick OFFLOAD false)         # Tunix offload_to_cpu: park inactive models in host RAM between phases
FORCE_FULL=$(pick FORCE_FULL 0)       # 1: every rollout generates exactly MAX_TOKENS (worst case for max-context tests)
VLLM_RESHARD_CHUNK=$(pick VLLM_RESHARD_CHUNK 16)  # weight sync in chunks of N tensors, deleting old vLLM buffers per chunk
# >0: compute logits/logprobs in sequence chunks of N tokens (rematerialized; saves HBM, costs some training time).
# Default 1024 from 4k-token completions (the [tokens, 152k-vocab] fp32 logits are what runs out of HBM first).
LOGPS_CHUNK=$(pick LOGPS_CHUNK $([ "$(pick MAX_TOKENS 1024)" -ge 4096 ] && echo 1024 || echo 0))
export LOGPS_CHUNK                   # tools/tunix_run.py gives the reference (KL) pass the same chunking
REMAT=$(pick REMAT 3)                # trainer activation remat: 3 = whole decoder layer, 2 = attention block, 1 = none
PREFIX_CACHE=$(pick PREFIX_CACHE true)  # vLLM prefix caching: a prompt's 8 rollouts share its prefill
XLA_OPT=$(pick XLA_OPT 1)            # 1: async collectives / collective-compute overlap XLA flags (v5e, as MaxText)
BETA=$(pick BETA "")                 # KL coefficient override (configs default to 0: no KL, no reference pass)
LR=$(pick LR "")                     # actor peak LR override; CRITIC_LR likewise (constant after WARMUP steps)
CRITIC_LR=$(pick CRITIC_LR "")
WARMUP=$(pick WARMUP 1)
DATA=$(pick DATA polaris)            # polaris (POLARIS-53K) | math (MATH L3-5 train, tools/math_data.py)
SAMPLING_KWARGS=$(pick SAMPLING_KWARGS "")   # extra vLLM SamplingParams, python dict body, e.g. 'stop_token_ids': [151645]
CRITIC_MICRO=${CRITIC_MICRO:-$((TRAIN_MICRO * GROUP))}   # PPO value pass: sequences per slice
export CRITIC_MICRO
# 1: correct the vLLM-vs-trainer log-prob mismatch with truncated importance sampling, w = min(pi_trainer/pi_vllm,
# TIS_CAP) per token (tools/tunix_run.py). Costs one trainer log-prob pass per step.
TIS=$(pick TIS 0); TIS_CAP=$(pick TIS_CAP 2.0)
export TIS TIS_CAP
VLLM_MAX_LEN=$(pick VLLM_MAX_LEN 0)  # >0: vLLM max_model_len (for evaluation longer than training rollouts)
export VLLM_MAX_LEN
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
if [ "$XLA_OPT" = 1 ]; then
  export LIBTPU_INIT_ARGS="${LIBTPU_INIT_ARGS:-} --xla_tpu_enable_async_collective_fusion=true \
--xla_tpu_enable_async_collective_fusion_fuse_all_gather=true --xla_tpu_enable_async_collective_fusion_multiple_steps=true \
--xla_tpu_overlap_compute_collective_tc=true --xla_enable_async_all_gather=true \
--xla_tpu_enable_data_parallel_all_reduce_opt=true --xla_tpu_data_parallel_opt_different_sized_ops=true"
fi
echo "worker=$WORKER algo=$ALGO host_mode=$HOST_MODE mode=$MODE model=$MODEL mesh=$MESH max_tokens=$MAX_TOKENS vllm_hbm=$VLLM_HBM logps_chunk=$LOGPS_CHUNK offload=$OFFLOAD remat=$REMAT prefix_cache=$PREFIX_CACHE xla_opt=$XLA_OPT train_micro=$TRAIN_MICRO data=$DATA"
# Fresh metrics for this launch (a crashed earlier attempt under the same run name must not leak in).
if [ "$HOST_MODE" = single ]; then rm -f "$LOG_ROOT/$RUN_NAME/eff-$WORKER.jsonl"
elif [ "$WORKER_ID" = 0 ]; then rm -f "$LOG_ROOT/$RUN_NAME/eff.jsonl"; fi

LORA_ARGS=()
SAMPLING=$SAMPLING_KWARGS
if [ "$FORCE_FULL" = 1 ]; then SAMPLING="${SAMPLING:+$SAMPLING, }'min_tokens': $MAX_TOKENS"; fi
case "$MODEL" in  # base models end their chat turn with <|im_end|>, which is not their EOS token
  *-Base) SAMPLING="${SAMPLING:+$SAMPLING, }'stop_token_ids': [151645, 151643]" ;;
esac
SAMPLING_ARGS=()
if [ -n "$SAMPLING" ]; then SAMPLING_ARGS=("rollout_config.rollout_vllm_sampling_kwargs={$SAMPLING}"); fi
if [ "$MODE" = lora ]; then
  export LORA_RANK=32 LORA_ALPHA=64.0   # tools/tunix_run.py merges LoRA into the weights synced to vLLM with alpha/rank
  LORA_MODULES='.*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj'
  LORA_ARGS=("actor_model_config.lora_config={'rank': $LORA_RANK, 'alpha': $LORA_ALPHA, 'module_path': '$LORA_MODULES'}")
  # PPO critic: same backbone + value head, its own LoRA adapter (full fine-tuning when MODE=full)
  if [ "$ALGO" = ppo ]; then
    LORA_ARGS+=("critic_model_config.lora_config={'rank': $LORA_RANK, 'alpha': $LORA_ALPHA, 'module_path': '$LORA_MODULES'}")
  fi
fi
OPT_ARGS=()
if [ -n "$LR" ]; then  # constant LR after a linear warmup (cosine from peak to peak)
  OPT_ARGS+=(rl_training_config.actor_optimizer_config.peak_value="$LR" rl_training_config.actor_optimizer_config.end_value="$LR"
             rl_training_config.actor_optimizer_config.warmup_steps="$WARMUP" rl_training_config.actor_optimizer_config.decay_steps="$NUM_BATCHES")
fi
if [ -n "$CRITIC_LR" ]; then
  OPT_ARGS+=(rl_training_config.critic_optimizer_config.peak_value="$CRITIC_LR" rl_training_config.critic_optimizer_config.end_value="$CRITIC_LR"
             rl_training_config.critic_optimizer_config.warmup_steps="$WARMUP" rl_training_config.critic_optimizer_config.decay_steps="$NUM_BATCHES")
fi
if [ -n "$BETA" ]; then OPT_ARGS+=("${ALGO}_config.beta=$BETA"); fi
if [ "$TIS" = 1 ]; then
  OPT_ARGS+=(rollout_config.return_logprobs=true)
  if [ "$ALGO" = ppo ]; then
    export ONPOLICY_OLD_LOGPS=0   # the trainer log-prob pass is needed for the weights
  else
    OPT_ARGS+=(grpo_config.use_rollout_logps=true grpo_config.sampler_is=token grpo_config.sampler_is_threshold="$TIS_CAP")
  fi
fi
DATA_MODULE=polaris_data
if [ "$DATA" = math ]; then DATA_MODULE=math_data; fi

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
  model_config.remat_config="$REMAT" \
  vllm_config.kwargs.enable_prefix_caching="$PREFIX_CACHE" \
  data_module="$DATA_MODULE" \
  "${LORA_ARGS[@]}" \
  "${SAMPLING_ARGS[@]}" \
  "${OPT_ARGS[@]}" \
  "$@"
