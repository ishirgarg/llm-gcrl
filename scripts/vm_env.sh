# Sourced on every TPU worker by the launch scripts. Defines where everything lives.
#
#   NFS  (shared, persistent, 100 GB quota): conda envs, code, small text logs
#   GCS  (same region as the VM, cheap):      model checkpoints, training checkpoints, profiles
#   Local disk (per worker, lost with VM):    re-downloadable caches (HF weights, datasets, pip, XLA)
#
# Nothing here ever copies data off GCP. Site-specific values (NFS_ROOT, BUCKET) come from the git-ignored
# site.env at the repo root (see site.env.example).

export REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source "$REPO/site.env"
export NFS_ROOT BUCKET
export CONDA_ROOT=$NFS_ROOT/miniconda3
export LOG_ROOT=$NFS_ROOT/logs

export CKPT_ROOT=$BUCKET/checkpoints     # training checkpoints
export MODEL_ROOT=$BUCKET/models         # converted base-model weights
export PROFILE_ROOT=$BUCKET/profiles     # xprof / tensorboard traces

export SCRATCH=$HOME/scratch
export HF_HOME=$SCRATCH/hf
export HF_DATASETS_CACHE=$SCRATCH/hf/datasets
export WANDB_DIR=$SCRATCH/wandb
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cache
mkdir -p "$SCRATCH" "$HF_HOME" "$WANDB_DIR" "$JAX_COMPILATION_CACHE_DIR" "$LOG_ROOT"

export WANDB_PROJECT=${WANDB_PROJECT:-llm-gcrl}

# wandb / HF tokens live only on the worker's local disk (never on NFS: it is world-readable)
if [ -f "$HOME/.tpu-secrets.env" ]; then
  source "$HOME/.tpu-secrets.env"
fi

activate_env() {
  source "$CONDA_ROOT/bin/activate" "$1"
}
export -f activate_env
