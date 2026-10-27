#!/bin/bash
# One-time SkyRL (JAX/TPU backend) install, run on worker 0 only. Everything lands on NFS,
# so all workers share the same venv. Usage (from laptop):
#   scripts/tpu.sh w0 "tmux new -d -s skyrl-install 'bash $REPO/scripts/run_on_worker.sh skyrl-install scripts/skyrl/install.sh'"
set -euxo pipefail

SKYRL_DIR=$NFS_ROOT/third_party/SkyRL
SKYRL_COMMIT=3a6e3b4e0e            # main as of 2026-10-22
COOKBOOK_DIR=$NFS_ROOT/third_party/tinker-cookbook
COOKBOOK_COMMIT=016468b0f214f30492f9f8eb001f9094970b3ad5  # pinned by SkyRL's own GSM8K CI test

activate_env llm-gcrl
pip install -q uv
export UV_CACHE_DIR=$SCRATCH/uv-cache   # cache on local disk; venv itself on NFS
export UV_LINK_MODE=copy                 # cache and venv are on different filesystems
export UV_PYTHON_PREFERENCE=only-system  # venv must use the NFS python, not a uv-managed one on local disk

mkdir -p "$NFS_ROOT/third_party"
[ -d "$SKYRL_DIR" ] || git clone https://github.com/NovaSky-AI/SkyRL.git "$SKYRL_DIR"
git -C "$SKYRL_DIR" fetch -q origin && git -C "$SKYRL_DIR" checkout -q "$SKYRL_COMMIT"
python3 "$REPO/tools/patch_skyrl.py" "$SKYRL_DIR"   # multi-host GCS training-checkpoint fix
[ -d "$COOKBOOK_DIR" ] || git clone https://github.com/thinking-machines-lab/tinker-cookbook.git "$COOKBOOK_DIR"
git -C "$COOKBOOK_DIR" fetch -q origin && git -C "$COOKBOOK_DIR" checkout -q "$COOKBOOK_COMMIT"

cd "$SKYRL_DIR"
uv sync --extra tpu --extra tinker --extra jax --extra gcp   # gcp: gs:// checkpoints_base
# Client side (GSM8K RL loop + wandb) in the same venv; CPU-only torch keeps the NFS footprint small.
# gcsfs: Flax/fsspec needs it for gs:// training checkpoints (see tools/patch_skyrl.py)
uv pip install --python .venv/bin/python -e "$COOKBOOK_DIR[math-rl,wandb]" datasets torch gcsfs \
  --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match

.venv/bin/python -c "import jax, skyrl, tinker, tinker_cookbook; print('jax', jax.__version__, 'tinker', tinker.__version__)"
du -sh "$SKYRL_DIR/.venv"
echo SKYRL_INSTALL_OK
