"""Idempotent local patch to SkyRL (pinned commit) for multi-host training checkpoints on GCS.

Bug: JaxBackendImpl.save_checkpoint/load_checkpoint pass a cloudpathlib GSPath to Flax/Orbax, which
resolves it to a per-process local cache dir (/tmp/tmpXXXX/<bucket>/...). On a multi-host slice every
process gets a different dir, Orbax's multi-process barrier waits for a directory the primary created
on another host, times out after 600 s and the whole job aborts at the first `save_every` step.
Fix: pass the plain "gs://..." string so all processes write the same checkpoint directly to GCS.

Bug 2 (skyrl/tx/utils/models.py, load_safetensors): jax.device_put(numpy, multi-host sharding) first runs
multihost_utils.assert_equal, which all-gathers the whole tensor onto every chip; for Qwen3-14B the embedding
doesn't fit (needs 5.80G with 4.48G free) and loading OOMs. Fix: jax.make_array_from_callback, which places
only each host's own shards (every host reads the same safetensors files, so the check is redundant).

Bug 3 (skyrl/tx/layers/stacked.py, StackedDecoderLayers): the stacked [num_layers, ...] buffers are made with
jnp.zeros on one chip and then device_put to their sharding; for Qwen3-30B-A3B the expert weights stack to
18 GiB on a single 15.75 GiB chip. Fix: create the zeros directly with the target sharding (jit out_shardings).

Bug 4 (skyrl/tx/utils/models.py, get_adapter_slice): the per-adapter LoRA slice assumes 3-D params
(adapter, in, rank) / (adapter, rank, out); MoE expert LoRA params have an extra experts axis, so the rank slice
hits the experts axis (128 -> 32) and saving / loading sampler weights for Qwen3-30B-A3B fails. Fix: index the
trailing axes with an Ellipsis.

Bug 5 (skyrl/tx/utils/models.py, save_safetensors): multi-host sampler-weight saves slice every expert of every
layer on device (128 x 48 x 3 slices for Qwen3-30B-A3B), keep them all alive, and only then all-gather to host:
a second on-device copy of the adapter that OOMs. Fix: all-gather each param to host first and slice in numpy
(the final all-gather is then skipped; it would wrongly concatenate host arrays across processes).

Usage: python patch_skyrl.py <SkyRL checkout>
"""
import pathlib
import sys

PATCHES_BACKEND = [
    ("            ckpt_dir=output_path,\n            step=0,\n            prefix=\"checkpoint_\",\n            overwrite=True,",
     "            ckpt_dir=str(output_path),  # llm-gcrl patch: shared gs:// path for multi-host\n"
     "            step=0,\n            prefix=\"checkpoint_\",\n            overwrite=True,"),
    ("            ckpt_dir=checkpoint_path,\n            target=self._extract_checkpoint_data(model_id),",
     "            ckpt_dir=str(checkpoint_path),  # llm-gcrl patch: shared gs:// path for multi-host\n"
     "            target=self._extract_checkpoint_data(model_id),"),
]

PATCHES_MODELS = [
    ("        return (adapter_index, slice(None), slice(None, rank))",
     "        return (adapter_index, ..., slice(None, rank))  # llm-gcrl patch: MoE expert params have an extra axis"),
    ("        return (adapter_index, slice(None, rank), slice(None))",
     "        return (adapter_index, ..., slice(None, rank), slice(None))  # llm-gcrl patch: MoE expert params"),
    ("            param.set_raw_value(jax.device_put(tensor.astype(param.dtype), param.sharding))",
     "            # llm-gcrl patch: place local shards only (device_put all-gathers numpy inputs multi-host)\n"
     "            host = np.asarray(tensor.astype(param.dtype))\n"
     "            param.set_raw_value(jax.make_array_from_callback(host.shape, param.sharding, lambda idx: host[idx]))"),
]

PATCHES_SAVE = [
    ("        if adapter_idx is not None:\n            param = param[adapter_idx]\n        if \"experts\" in path:\n"
     "            for i in range(config.get_num_experts()):\n                tensors[get_expert_key(path, i)] = param[i, :, :].T",
     "        if adapter_idx is not None:\n            param = param[adapter_idx]\n"
     "        if jax.process_count() > 1:  # llm-gcrl patch: to host first, slice in numpy (no on-device copies)\n"
     "            from jax.experimental import multihost_utils as _mhu\n"
     "            param = np.asarray(_mhu.process_allgather(param, tiled=True))\n"
     "        if \"experts\" in path:\n"
     "            for i in range(config.get_num_experts()):\n                tensors[get_expert_key(path, i)] = param[i, :, :].T"),
    ("    if jax.process_count() > 1:\n        from jax.experimental import multihost_utils\n\n"
     "        tensors = {k: multihost_utils.process_allgather(v, tiled=True) for k, v in tensors.items()}",
     "    if False:  # llm-gcrl patch: params were gathered to host above\n        from jax.experimental import multihost_utils\n\n"
     "        tensors = {k: multihost_utils.process_allgather(v, tiled=True) for k, v in tensors.items()}"),
]

PATCHES_STACKED = [
    ("                stacked = jax.device_put(jnp.zeros(stacked_shape, arr.dtype), NamedSharding(mesh, new_spec))",
     "                # llm-gcrl patch: allocate already sharded (an unsharded jnp.zeros can exceed one chip's HBM)\n"
     "                stacked = jax.jit(functools.partial(jnp.zeros, stacked_shape, arr.dtype),\n"
     "                                  out_shardings=NamedSharding(mesh, new_spec))()"),
]


def _apply(path, patches):
  src = path.read_text()
  for old, new in patches:
    if new in src:
      continue
    if src.count(old) != 1:
      sys.exit(f"patch_skyrl: expected exactly one match in {path}; SkyRL changed?\n{old}")
    src = src.replace(old, new)
  path.write_text(src)
  print(f"patch_skyrl: {path} patched")


def main():
  root = pathlib.Path(sys.argv[1]) / "skyrl"
  _apply(root / "backends" / "jax.py", PATCHES_BACKEND)
  _apply(root / "tx" / "utils" / "models.py", PATCHES_MODELS + PATCHES_SAVE)
  _apply(root / "tx" / "layers" / "stacked.py", PATCHES_STACKED)


if __name__ == "__main__":
  main()
