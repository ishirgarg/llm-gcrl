# Source on a worker to make libtpu treat THIS host's 4 v5e chips as a standalone single-host TPU, so each host
# of the v5e-16 slice can run an independent JAX program (no jax.distributed, no cross-host collectives).
# Same mechanism vLLM tpu-inference uses to "isolate each host as its own JAX cluster" for Ray multi-host.
export TPU_PROCESS_ADDRESSES=localhost:${TPU_PROCESS_PORT:-8476}
export TPU_PROCESS_PORT=${TPU_PROCESS_PORT:-8476}
export CLOUD_TPU_TASK_ID=0
export TPU_PROCESS_BOUNDS=1,1,1
export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1   # a v5e host is a 2x2 block of chips
export TPU_VISIBLE_CHIPS=0,1,2,3
export SINGLE_HOST=1
# Separate XLA compilation cache: executables compiled for this standalone-host topology must never be reused
# by multi-host jobs (the cache key ignores process bounds; reuse aborts in libtpu's tpu_topology.cc).
export JAX_COMPILATION_CACHE_DIR=$SCRATCH/jax_cache_single
mkdir -p "$JAX_COMPILATION_CACHE_DIR"
