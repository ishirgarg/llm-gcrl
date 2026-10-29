"""Run Tunix's vLLM (tpu-inference) rollout on a plain multi-host TPU slice: one vLLM replica per host inside a
single multi-controller JAX job (what Pathways otherwise provides).

Without this, Tunix hands vLLM every device of the global rollout mesh, and each host's in-process engine fails
on devices it can't address (KeyError in tpu_worker.init_device). Patches:

  0. CPU staging: jax.devices("cpu") returns this host's CPU devices (tpu-inference loads weights via
     jax.devices("cpu")[0], which is otherwise process 0's CPU).
  1. Engine devices: each process's vLLM gets only its local chips (TP = local chip count, DP = 1). Exception:
     on the process that owns the slice-origin chips, libtpu rejects a program spanning exactly its 4 chips
     ("Unexpected device_id 4 in replica group (0,1,2,3)"; any 2-3 chip subset works, other hosts are fine),
     so that replica uses its first ORIGIN_VLLM_CHIPS chips. Default 0: the origin process runs no vLLM engine
     (its chips only train; a 2-chip replica can't hold an 8B+ model next to the trainer).
  2. Weight sync (update_params / load_checkpoint): every process first gathers the updated weights to host RAM
     (one collective per leaf over the trainer's state, identical on all processes), then each engine loads them
     locally (reshard_pytree onto its own vLLM mesh, no cross-host traffic); a process without an engine stops
     after the gather.
  3. Generate: every process holds the same prompt batch; each generates a contiguous slice proportional to its
     replica's chip count, then tokens, logprobs and prompt tokens are all-gathered so every process returns
     the full, identically ordered batch.

Install before Tunix builds the RL engine: `import tunix_multihost_vllm; tunix_multihost_vllm.install()`.
"""
import functools
import os

import jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from tunix.generate import base_sampler
from tunix.generate import utils as gen_utils
from tunix.generate import vllm_sampler
from tunix.rl import reshard


def _local_devices():
  return sorted(jax.local_devices(), key=lambda d: d.id)


def _owns_origin():
  return any(d.id == 0 for d in jax.local_devices())


def _vllm_devices():
  """Chips this process's vLLM replica uses (see the origin-host exception in the module docstring)."""
  devs = _local_devices()
  if _owns_origin():
    devs = devs[: int(os.environ.get("ORIGIN_VLLM_CHIPS", "0"))]
  # NO_VLLM_PROCESSES="5,6": these processes run no vLLM engine either (they only train), e.g. to route
  # around a process whose engine fails.
  skip = {int(p) for p in os.environ.get("NO_VLLM_PROCESSES", "").split(",") if p.strip()}
  if jax.process_index() in skip:
    devs = []
  return devs


def _has_engine():
  return len(_vllm_devices()) > 0


# ---------------------------------------------------------------------------------------------- 0. engine-less process
_orig_init = vllm_sampler.VllmSampler.__init__


def _init(self, tokenizer, config):
  if _has_engine():
    return _orig_init(self, tokenizer, config)
  # Stub: tokenizer/config only. This process trains but generates nothing (zero prompts in _call).
  self.tokenizer = tokenizer
  if not isinstance(tokenizer, vllm_sampler.tok_adapter.TokenizerAdapter):
    self.tokenizer = vllm_sampler.tok_adapter.TokenizerAdapter(tokenizer)
  self.config = config
  self._driver = None
  self.llm = None
  self.args = {}


def _mesh(self):
  if not _has_engine():
    return Mesh(np.array(_local_devices()), ("model",))
  return _orig_mesh.fget(self)


def _transformer(self):
  return _orig_transformer.fget(self) if _has_engine() else None


_orig_mesh = vllm_sampler.VllmSampler.mesh
_orig_transformer = vllm_sampler.VllmSampler.transformer


# ---------------------------------------------------------------------------------------------- 1. engine devices
_orig_vllm_config = vllm_sampler.VllmSampler._vllm_config


def _vllm_config(self, config):
  args = _orig_vllm_config(self, config)
  local = [d.id for d in _vllm_devices()]
  args["tensor_parallel_size"] = len(local)
  args["data_parallel_size"] = 1
  additional = dict(args.get("additional_config") or {})
  sharding = dict(additional.get("sharding") or {})
  strategy = dict(sharding.get("sharding_strategy") or {})
  strategy["device_indexes"] = local
  sharding["sharding_strategy"] = strategy
  additional["sharding"] = sharding
  args["additional_config"] = additional
  return args


# ---------------------------------------------------------------------------------------------- 2. weight sync
@functools.lru_cache(maxsize=None)
def _host_local_spec(mesh, shape):
  """PartitionSpec splitting one dim across a mesh axis that lies within each host (replicated across hosts)."""
  for axis in mesh.axis_names:
    devs = np.asarray(mesh.devices)
    # an axis is host-local if moving along it never changes the process
    moved = np.moveaxis(devs, mesh.axis_names.index(axis), 0)
    if not all(len({d.process_index for d in moved[(slice(None),) + idx]}) == 1
               for idx in np.ndindex(moved.shape[1:])):
      continue
    size = mesh.shape[axis]
    for dim, n in enumerate(shape):
      if n % size == 0:
        return P(*([None] * dim + [axis]))
  return None


def _to_host(x):
  """Full host copy of a (possibly global) array on every process. Collective: same leaf order on all processes.

  multihost_utils.process_allgather replicates the whole tensor on every chip first (2.9 GiB for Qwen3-14B's
  embedding: OOM next to vLLM). Instead reshard so each host's chips hold one quarter each (replicated across
  hosts) and assemble the host copy from the local shards.
  """
  if not isinstance(x, jax.Array):
    return x
  if x.is_fully_addressable:
    return np.asarray(x)
  mesh = getattr(x.sharding, "mesh", None)
  spec = _host_local_spec(mesh, x.shape) if isinstance(x.sharding, NamedSharding) else None
  if spec is not None:
    y = jax.jit(lambda a: a, out_shardings=NamedSharding(mesh, spec))(x)
    host = np.empty(x.shape, dtype=x.dtype)
    covered = 0
    seen = set()
    for shard in y.addressable_shards:
      key = tuple((s.start, s.stop) for s in shard.index)
      if key in seen:
        continue
      seen.add(key)
      data = np.asarray(shard.data)
      host[shard.index] = data
      covered += data.size
    del y
    if covered == host.size:
      return host
  return np.asarray(multihost_utils.process_allgather(x, tiled=True))


_orig_update_params = vllm_sampler.VllmSampler.update_params


def _update_params(self, updated_weights, filter_types=None):
  host_weights = jax.tree.map(_to_host, updated_weights)
  if not _has_engine():
    return None
  return _orig_update_params(self, host_weights, filter_types)


def _is_process_local(sharding):
  if not isinstance(sharding, NamedSharding):
    return False
  devs = list(sharding.mesh.devices.flat)
  return len(devs) < jax.device_count() and all(d.process_index == jax.process_index() for d in devs)


@functools.lru_cache(maxsize=None)
def _all_devices_mesh():
  return Mesh(np.array(jax.devices()), ("_all",))  # identical on every process


def _to_local(src, tgt):
  tgt_sharding = tgt.sharding if isinstance(tgt, jax.Array) else tgt
  if isinstance(src, jax.Array) and not src.is_fully_addressable:
    # Spread across hosts: replicate on every chip first. A collective, but every process reaches it for the same
    # leaf (sources have the same kind on all processes), one leaf at a time so the extra memory is transient.
    src = jax.device_put(src, NamedSharding(_all_devices_mesh(), P())).addressable_shards[0].data
  # Now a full local copy (or host data): lay it out on this host's vLLM mesh, no cross-host traffic.
  return jax.device_put(src, tgt_sharding)


_orig_reshard_pytree = reshard.reshard_pytree


def _reshard_pytree(source, target, *args, **kwargs):
  is_leaf = lambda x: isinstance(x, jax.sharding.Sharding)
  leaves = jax.tree.leaves(target, is_leaf=is_leaf)
  first = leaves[0] if leaves else None
  first_sharding = first.sharding if isinstance(first, jax.Array) else first
  if not _is_process_local(first_sharding):
    return _orig_reshard_pytree(source, target, *args, **kwargs)
  return jax.tree.map(_to_local, source, target, is_leaf=is_leaf)


_orig_transfer = gen_utils.transfer_state_with_mappings


def _transfer_state_with_mappings(src_state, dst_state, *args, reshard_fn=None, **kwargs):
  """Free each replaced vLLM weight before the new one is laid out on HBM.

  Tunix maps every new value into dst_state first, then reshards. Its delete_dst_buffers step receives the
  snapshot of target *shardings*, not arrays, so it deletes nothing and the old weights stay alive until the
  sync returns: two copies of the vLLM weights at once (+7.4 GiB/chip for 14B on a 4-chip replica). Here, at the
  first reshard call (all values mapped, nothing resharded yet), delete exactly the old arrays that were replaced.
  """
  if reshard_fn is None:
    return _orig_transfer(src_state, dst_state, *args, reshard_fn=reshard_fn, **kwargs)
  old = [x for x in jax.tree.leaves(dst_state) if isinstance(x, jax.Array)]

  def reshard_and_free(*rargs, **rkwargs):
    if old:
      live = {id(x) for x in jax.tree.leaves(dst_state)}
      freed = 0
      for x in old:
        if id(x) not in live and not x.is_deleted():
          freed += x.nbytes
          x.delete()
      print(f"[tunix_multihost_vllm] weight sync: freed {freed / 2**30:.2f} GiB of replaced vLLM weights "
            f"({len(old)} old arrays)", flush=True)
      old.clear()
    return reshard_fn(*rargs, **rkwargs)

  return _orig_transfer(src_state, dst_state, *args, reshard_fn=reshard_and_free, **kwargs)


# ---------------------------------------------------------------------------------------------- 3. generate
_orig_call = vllm_sampler.VllmSampler.__call__


@functools.lru_cache(maxsize=None)
def _replica_weights():  # chips per process's replica (0 = no engine)
  return tuple(int(w) for w in np.asarray(multihost_utils.process_allgather(np.array([len(_vllm_devices())]))).reshape(-1))


def _split_counts(n):
  """Contiguous per-process prompt counts proportional to replica chip counts (largest remainder)."""
  w = np.array(_replica_weights(), dtype=np.float64)
  exact = n * w / w.sum()
  counts = np.floor(exact).astype(int)
  for i in np.argsort(-(exact - counts))[: n - counts.sum()]:
    counts[i] += 1
  return counts


def _gather_ragged(seqs, counts, dtype, pad_value):
  """All-gather per-process lists of 1-D sequences (process p holds counts[p] of them); returns the global list."""
  rows = int(max(counts))
  lens = np.zeros(rows, dtype=np.int32)
  lens[: len(seqs)] = [len(s) for s in seqs]
  width = max(int(multihost_utils.process_allgather(np.array([lens.max(initial=0)])).max()), 1)
  buf = np.full((rows, width), pad_value, dtype=dtype)
  for i, s in enumerate(seqs):
    buf[i, : len(s)] = np.asarray(s, dtype=dtype)
  all_buf = np.asarray(multihost_utils.process_allgather(buf)).reshape(len(counts), rows, width)
  all_lens = np.asarray(multihost_utils.process_allgather(lens)).reshape(len(counts), rows)
  return [all_buf[p, i, : all_lens[p, i]] for p in range(len(counts)) for i in range(counts[p])]


def _call(self, input_strings, max_generation_steps, max_prompt_length=None, **kwargs):
  nproc = jax.process_count()
  if nproc == 1:
    return _orig_call(self, input_strings, max_generation_steps, max_prompt_length=max_prompt_length, **kwargs)
  if isinstance(input_strings, str):
    input_strings = [input_strings]
  counts = _split_counts(len(input_strings))
  p = jax.process_index()
  start = int(counts[:p].sum())
  if counts[p]:
    out = _orig_call(self, input_strings[start:start + counts[p]], max_generation_steps,
                     max_prompt_length=max_prompt_length, **kwargs)
  else:  # engine-less process: empty local slice, but it still takes part in every gather below
    out = base_sampler.SamplerOutput(text=[], logits=None, tokens=[], padded_prompt_tokens=np.zeros((0, 0), np.int32),
                                     logprobs=[], routed_experts=None)

  tokens = _gather_ragged(out.tokens, counts, np.int32, 0)
  logprobs = None
  if self.config.return_logprobs:  # same branch on every process (not out.logprobs, which differs when empty)
    logprobs = [list(map(float, x)) for x in _gather_ragged(out.logprobs, counts, np.float32, 0.0)]
  # Left-padded prompts: right-align to a common width, pad rows to the largest count, then gather.
  prompts = np.asarray(out.padded_prompt_tokens)
  width = int(multihost_utils.process_allgather(np.array([prompts.shape[1]])).max())
  pad_id = self.tokenizer.pad_id()
  full = np.full((int(max(counts)), width), pad_id, dtype=np.int32)
  full[: prompts.shape[0], width - prompts.shape[1]:] = prompts
  gathered = np.asarray(multihost_utils.process_allgather(full)).reshape(len(counts), -1, width)
  prompts = np.concatenate([gathered[q, : counts[q]] for q in range(len(counts))], axis=0)
  text = [self.tokenizer.decode(t.tolist()) for t in tokens]
  return base_sampler.SamplerOutput(text=text, logits=None, tokens=tokens, padded_prompt_tokens=prompts,
                                    logprobs=logprobs, routed_experts=None)


_orig_jax_devices = jax.devices


def _jax_devices(backend=None):
  # tpu-inference stages weights on jax.devices("cpu")[0], which in a multi-controller job is process 0's CPU;
  # on other hosts those arrays aren't addressable. CPU staging must use this host's own CPU devices.
  if backend == "cpu":
    return jax.local_devices(backend="cpu")
  return _orig_jax_devices(backend)


# ---------------------------------------------------------------------------------------------- 4. PPO metrics
class _MaskedArrayProxy:
  """np.ma with masked_array() gathering cross-host jax.Arrays to the host first."""

  def __getattr__(self, name):
    return getattr(np.ma, name)

  def masked_array(self, data, *args, **kwargs):
    return np.ma.masked_array(_to_host(data), *args, **{k: _to_host(v) for k, v in kwargs.items()})


class _NumpyProxy:
  """Stand-in for `np` inside tunix.rl.ppo.ppo_learner.

  PPO logs advantage / return / value stats with np.ma.masked_array on critic outputs that live on the
  16-chip trainer mesh (not fully addressable), which raises in multi-host mode. Gathering them first is a
  collective, which is fine: every process runs the same metric code in the same order.
  """
  ma = _MaskedArrayProxy()

  def __getattr__(self, name):
    return getattr(np, name)


def _patch_ppo_metrics():
  try:
    from tunix.rl.ppo import ppo_learner  # pylint: disable=g-import-not-at-top
  except ImportError:
    return
  ppo_learner.np = _NumpyProxy()


def _patch_shard_input():
  """Shard trainer inputs from full per-process copies instead of concatenating them.

  Every process holds the whole rollout batch (generate gathers it everywhere), but Tunix's
  sharding_utils.shard_input builds the global array with make_array_from_process_local_data, which treats each
  process's array as its *slice*: the global batch came out 8x duplicated on 8 hosts (same gradient, 8x the
  compute and memory in every training micro-step and logprob pass). It also skipped sharding entirely when any
  leaf was already global (e.g. GRPO with KL), so the bug showed only in some configurations. Build each global
  array from the local copy instead (each process supplies its addressable shards); leave global leaves as they are.
  """
  from jax.interpreters import pxla  # pylint: disable=g-import-not-at-top
  from tunix.sft import sharding_utils  # pylint: disable=g-import-not-at-top
  from jax.sharding import PartitionSpec  # pylint: disable=g-import-not-at-top

  def shard_input(input_data, data_sharding_axis):
    mesh = pxla.thread_resources.env.physical_mesh
    if mesh.empty:
      return input_data
    pspec = PartitionSpec(*data_sharding_axis)

    def put(x):
      if not isinstance(x, (np.ndarray, jax.Array)):
        return x
      if isinstance(x, jax.Array) and not x.is_fully_addressable:
        return x  # already a global array
      sharding = sharding_utils.get_sharding(x, mesh=mesh, pspec=pspec)
      if isinstance(x, jax.Array) and x.sharding == sharding:
        return x
      host = np.asarray(x)
      return jax.make_array_from_callback(host.shape, sharding, lambda idx: host[idx])

    with jax.transfer_guard("allow"):
      return jax.tree.map(put, input_data)

  sharding_utils.shard_input = shard_input


def _patch_weight_loading():
  """Load safetensors weights shard by shard on multi-host meshes.

  Tunix's optimized loader calls jax.device_put(full host tensor, global sharding) per weight. Across processes
  that runs as a compiled program whose temporaries grow with the model (19.7 GiB for Qwen3-8B, 24.6 GiB for 14B:
  OOM before training starts). Every host already holds the full tensor (memory-mapped), so build each global
  array with make_array_from_callback: each process copies only its addressable shards from the host array.
  """
  from tunix.models import safetensors_loader  # pylint: disable=g-import-not-at-top

  class _JaxProxy:

    def __getattr__(self, name):
      return getattr(jax, name)

    @staticmethod
    def device_put(x, device=None, *args, **kwargs):
      if (isinstance(x, np.ndarray) and hasattr(device, "is_fully_addressable")
          and not device.is_fully_addressable and not args and not kwargs):
        return jax.make_array_from_callback(x.shape, device, lambda idx: np.asarray(x[idx]))
      return jax.device_put(x, device, *args, **kwargs)

  safetensors_loader.jax = _JaxProxy()


def install():
  if jax.process_count() == 1:
    return
  _patch_shard_input()
  _patch_weight_loading()
  jax.devices = _jax_devices
  vllm_sampler.VllmSampler.__init__ = _init
  vllm_sampler.VllmSampler.mesh = property(_mesh)
  vllm_sampler.VllmSampler.transformer = property(_transformer)
  vllm_sampler.VllmSampler.update_params = _update_params
  vllm_sampler.VllmSampler._vllm_config = _vllm_config
  vllm_sampler.VllmSampler.__call__ = _call
  reshard.reshard_pytree = _reshard_pytree
  gen_utils.transfer_state_with_mappings = _transfer_state_with_mappings
  _patch_ppo_metrics()
  print(f"[tunix_multihost_vllm] one vLLM replica per host: process {jax.process_index()}/{jax.process_count()}, "
        f"vLLM devices {[d.id for d in _vllm_devices()]} of local {[d.id for d in _local_devices()]}", flush=True)
