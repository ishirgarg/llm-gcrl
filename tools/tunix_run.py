"""Tunix CLI launcher (GRPO or PPO) with the shared eff/* metrics, for either deployment mode:

  SINGLE_HOST=1 (scripts/single_host_env.sh): every host is an independent 4-chip TPU; no jax.distributed.
  otherwise: the whole multi-host slice runs one program; jax.distributed is initialised before Tunix starts
  (tunix.cli assumes single-host or Pathways).

Usage: TUNIX_ALGO=grpo|ppo python tools/tunix_run.py <base_config.yaml> override_config_file=... key=value ...
"""
import os
import sys

import jax

if os.environ.get("SINGLE_HOST") != "1":
  jax.distributed.initialize()

from absl import app  # pylint: disable=g-import-not-at-top
from tpu_utils import prevent_cross_region  # pylint: disable=g-import-not-at-top
import tunix_eff  # pylint: disable=g-import-not-at-top

if os.environ.get("TUNIX_ALGO", "grpo") == "ppo":
  from tunix.cli import ppo_main as entry  # pylint: disable=g-import-not-at-top
  if os.environ.get("PPO_GROUP"):
    # Tunix's PPOLearner hard-codes one rollout per prompt; the base learner already repeats each prompt
    # _num_generations() times before generation, and GAE advantages are per sequence, so PPO_GROUP=8 gives
    # 8 rollouts per prompt (micro-batches stay one prompt = PPO_GROUP sequences, as in GRPO).
    from tunix.rl.ppo import ppo_learner  # pylint: disable=g-import-not-at-top
    _ppo_group = int(os.environ["PPO_GROUP"])
    ppo_learner.PPOLearner._num_generations = lambda self: _ppo_group  # pylint: disable=protected-access
    # PPO also never scales its rollout / logprob micro-batches by the group size (GRPO multiplies both by
    # num_generations). Unscaled, 16 prompts x 8 rollouts are generated in 8 sequential chunks of 16 and the
    # old-logprob pass runs one sequence at a time. Scale them for the duration of the rollout phase.
    _orig_gen_adv = ppo_learner.PPOLearner._generate_and_compute_advantage  # pylint: disable=protected-access

    def _gen_adv_grouped(self, *args, **kwargs):
      saved = self._rollout_micro_batch_size, self._compute_logps_micro_batch_size  # pylint: disable=protected-access
      self._rollout_micro_batch_size = saved[0] * _ppo_group if saved[0] else saved[0]
      self._compute_logps_micro_batch_size = saved[1] * _ppo_group if saved[1] else saved[1]
      try:
        return _orig_gen_adv(self, *args, **kwargs)
      finally:
        self._rollout_micro_batch_size, self._compute_logps_micro_batch_size = saved

    ppo_learner.PPOLearner._generate_and_compute_advantage = _gen_adv_grouped  # pylint: disable=protected-access

  # Tunix runs the critic's value pass on the whole rollout batch at once (every other pass is micro-batched),
  # so its activations scale with the total rollout count and it is the first thing to OOM at long context.
  # Run it in slices of CRITIC_MICRO sequences (default: one prompt's group) and concatenate; values are
  # per-sequence, so the result is identical.
  from tunix.rl import rl_cluster as _rl_cluster_v  # pylint: disable=g-import-not-at-top
  import jax.numpy as _jnp  # pylint: disable=g-import-not-at-top
  _critic_micro = int(os.environ.get("CRITIC_MICRO") or os.environ.get("PPO_GROUP") or 8)
  _orig_get_values = _rl_cluster_v.RLEngine.get_values

  def _get_values_micro(self, prompt_tokens, completion_tokens, pad_id, eos_id):
    n = prompt_tokens.shape[0]
    if n <= _critic_micro:
      return _orig_get_values(self, prompt_tokens, completion_tokens, pad_id, eos_id)
    return _jnp.concatenate([
        _orig_get_values(self, prompt_tokens[i:i + _critic_micro], completion_tokens[i:i + _critic_micro],
                         pad_id, eos_id)
        for i in range(0, n, _critic_micro)], axis=0)

  _rl_cluster_v.RLEngine.get_values = _get_values_micro
  if os.environ.get("SINGLE_HOST") != "1":
    # PPO takes old-policy logps from the rollout engine; with per-host vLLM replicas (tunix_multihost_vllm)
    # that returns an empty array. Compute them with the trainer's actor instead (as verl does); with one
    # on-policy update per batch they equal the current policy's logps. Patched before tunix_eff wraps it.
    from tunix.rl import rl_cluster as _rl_cluster  # pylint: disable=g-import-not-at-top

    def _old_logps_from_actor(self, prompt_tokens, completion_tokens, micro_batch_size=None):
      return self.get_actor_per_token_logps(
          prompt_tokens, completion_tokens, pad_id=self.rollout.pad_id(), eos_id=self.rollout.eos_id(),
          micro_batch_size=micro_batch_size)

    _rl_cluster.RLEngine.get_old_per_token_logps = _old_logps_from_actor

  if os.environ.get("ONPOLICY_OLD_LOGPS", "1") == "1":
    # With one policy update per rollout batch (num_iterations 1, mini_batch = batch) every micro-step's loss
    # runs on the pre-update weights, so the old-policy logps equal the current ones exactly. Skip that extra
    # actor forward pass over the whole batch and use stop_gradient(current logps) in the loss instead (what
    # Tunix's GRPO already does when it has no old logps). Only without KL (the KL term needs them).
    import dataclasses as _dc  # pylint: disable=g-import-not-at-top
    from tunix.rl import algo_core as _algo_core  # pylint: disable=g-import-not-at-top
    from tunix.rl import function_registry as _fn_registry  # pylint: disable=g-import-not-at-top

    _orig_gen_adv_op = ppo_learner.PPOLearner._generate_and_compute_advantage  # pylint: disable=protected-access

    def _gen_adv_onpolicy(self, *args, **kwargs):
      if self.algo_config.num_iterations != 1 or self.algo_config.beta != 0.0:
        return _orig_gen_adv_op(self, *args, **kwargs)
      self.rl_engine.get_old_per_token_logps = lambda *a, **k: None  # instance attr shadows the class method
      try:
        return _orig_gen_adv_op(self, *args, **kwargs)
      finally:
        del self.rl_engine.get_old_per_token_logps

    ppo_learner.PPOLearner._generate_and_compute_advantage = _gen_adv_onpolicy  # pylint: disable=protected-access

    class _OnPolicyLogps:
      """Stands in for missing old logps: `logps - old` evaluates to logps - stop_gradient(logps) (ratio 1)."""

      def __rsub__(self, logps):
        return logps - jax.lax.stop_gradient(logps)

    _orig_ppo_loss = _algo_core.ppo_policy_loss_fn

    def _ppo_loss_onpolicy(model, train_example, algo_config, pad_id, eos_id, **kwargs):
      if train_example.old_per_token_logps is None:
        train_example = _dc.replace(train_example, old_per_token_logps=_OnPolicyLogps())
      return _orig_ppo_loss(model, train_example, algo_config, pad_id, eos_id, **kwargs)

    _fn_registry.register_policy_loss_fn("ppo")(_ppo_loss_onpolicy)
else:
  from tunix.cli import grpo_main as entry  # pylint: disable=g-import-not-at-top
  import dataclasses as _dc  # pylint: disable=g-import-not-at-top
  from tunix.rl.grpo import grpo_learner as _grpo_learner  # pylint: disable=g-import-not-at-top
  if "epsilon_high" not in {f.name for f in _dc.fields(_grpo_learner.GRPOConfig)}:
    # The installed GRPOConfig has no epsilon_high field, but its loss reads one when present (clip-higher):
    # accept grpo_config.epsilon_high and set it as an attribute.
    _orig_grpo_cfg_init = _grpo_learner.GRPOConfig.__init__

    def _grpo_cfg_init(self, *args, epsilon_high=None, **kwargs):
      _orig_grpo_cfg_init(self, *args, **kwargs)
      object.__setattr__(self, "epsilon_high", self.epsilon if epsilon_high is None else float(epsilon_high))

    _grpo_learner.GRPOConfig.__init__ = _grpo_cfg_init

if int(os.environ.get("LOGPS_CHUNK") or 0) > 0:
  # Tunix chunks the actor's logits when compute_logps_chunk_size is set, but its reference-model pass (KL term)
  # ignores it and materialises the full [batch, length, vocab] logits, which is what runs out of HBM first at
  # long context with KL on. Give the reference pass the same chunk size.
  from tunix.rl import common as _common  # pylint: disable=g-import-not-at-top
  from tunix.rl.inference import inference_worker as _inference_worker  # pylint: disable=g-import-not-at-top
  _ref_chunk = int(os.environ["LOGPS_CHUNK"])

  class _ChunkedCommon:

    def __getattr__(self, name):
      return getattr(_common, name)

    @staticmethod
    def compute_per_token_logps(*args, **kwargs):
      kwargs.setdefault("chunk_size", _ref_chunk)
      return _common.compute_per_token_logps(*args, **kwargs)

  _inference_worker.common = _ChunkedCommon()

def _patch_lora_merged_sync():
  """Sync LoRA training to vLLM as merged full weights.

  With LoRA, Tunix's RLEngine.sync_weights sends only the LoRAParam tensors; vLLM loads them only if it was built
  with LoRA support (vllm_config.lora_config + LoRA key mappings). Without that they are dropped ("No mapping for
  source key: ...w_lora_a") and rollouts keep using the base model. Instead send W + (alpha / rank) * A @ B for
  every LoRA-wrapped weight through the regular full-weight path (the same math as Tunix's
  save_lora_merged_model_as_safetensors), so vLLM always samples from the current policy.
  """
  import flax.nnx as nnx  # pylint: disable=g-import-not-at-top
  import jax  # pylint: disable=g-import-not-at-top,redefined-outer-name,reimported
  from tunix.rl import rl_cluster as _rc  # pylint: disable=g-import-not-at-top
  from tunix.sft import utils as sft_utils  # pylint: disable=g-import-not-at-top

  scale = float(os.environ.get("LORA_ALPHA", 64.0)) / float(os.environ.get("LORA_RANK", 32))
  orig_sync = _rc.RLEngine.sync_weights

  def merged_param_state(model):
    """Flat {path tuple: array} of all nnx.Param weights, with LoRA deltas merged in (sync accepts a flat dict)."""
    flat = {tuple(k): (v.value if hasattr(v, "value") else v) for k, v in nnx.to_flat_state(nnx.state(model, nnx.Param))
            if not isinstance(v, nnx.LoRAParam) and "lora" not in str(k[-1])}  # LoRAParam subclasses Param
    pairs = {}
    for path, value in nnx.iter_graph(model):
      if isinstance(value, nnx.LoRAParam):
        name = str(path[-1])
        base = name.replace("_lora_a", "").replace("_lora_b", "")
        pairs.setdefault(tuple(path[:-1]) + (base,), {})["a" if "lora_a" in name else "b"] = (
            value.value if hasattr(value, "value") else value)
    norm = {tuple(str(p) for p in k): k for k in flat}
    merged = 0
    for base_path, ab in pairs.items():
      key = norm.get(tuple(str(p) for p in base_path))
      if key is None or "a" not in ab or "b" not in ab:
        raise RuntimeError(f"LoRA merge: no base weight for {base_path}")
      w, a, b = flat[key], ab["a"], ab["b"]
      r = a.shape[-1]
      delta = (a.reshape(-1, r) @ b.reshape(r, -1)).reshape(w.shape) * scale
      flat[key] = w + delta.astype(w.dtype)
      merged += 1
    # dotted string keys: tuple keys mix str/int, which jax cannot sort when walking the dict as a pytree
    return {".".join(str(p) for p in k): v for k, v in flat.items()}, merged

  def sync_weights_merged(self):
    # Wrap the installed sync_weights (keeps its bookkeeping: step counter, anchor state, gc) and only swap the
    # payload it hands to the rollout: the merged full weights instead of the LoRA tensors.
    if not sft_utils.is_lora_enabled(self.actor_trainer.model):
      return orig_sync(self)
    rollout = self.rollout
    orig_update = rollout.update_params

    def update_with_merged(params, filter_types=None):  # pylint: disable=unused-argument
      merged_state, n = merged_param_state(self.actor_trainer.model)
      if not getattr(self, "_lora_merge_logged", False):
        print(f"[tunix_run] LoRA weight sync: merged {n} LoRA layers into base weights (scale {scale})", flush=True)
        self._lora_merge_logged = True
      return orig_update(merged_state, (nnx.Param,))

    rollout.update_params = update_with_merged
    try:
      return orig_sync(self)
    finally:
      del rollout.update_params  # back to the class method

  _rc.RLEngine.sync_weights = sync_weights_merged


_patch_lora_merged_sync()  # before tunix_eff wraps sync_weights, so its timing includes the merge
if os.environ.get("EVAL_SETS"):
  import rl_eval  # pylint: disable=g-import-not-at-top
  rl_eval.install()  # before tunix_eff, so evaluation generations are not counted as rollouts
tunix_eff.install()
if os.environ.get("EVAL_SETS"):
  rl_eval.install_hooks()  # after tunix_eff, so evaluation time is not counted as weight-sync time


def _hf_from_cache(model_id, model_download_path):  # pylint: disable=unused-argument
  # Tunix's hf_pipeline copies every file into model_download_path (a second full copy on local disk);
  # load from the worker-local HF cache instead, the same copy vLLM and SkyRL use.
  from huggingface_hub import snapshot_download  # pylint: disable=g-import-not-at-top
  return snapshot_download(model_id, ignore_patterns=["original/*"])


from tunix.oss import utils as _oss_utils  # pylint: disable=g-import-not-at-top
_oss_utils.hf_pipeline = _hf_from_cache
if os.environ.get("SINGLE_HOST") != "1":
  import tunix_multihost_vllm  # pylint: disable=g-import-not-at-top
  tunix_multihost_vllm.install()  # one vLLM replica per host inside the multi-host job


def _gcs_args():
  return [a.split("=", 1)[1] for a in sys.argv[2:] if "=" in a and a.split("=", 1)[1].startswith("gs://")]


def main(argv):
  prevent_cross_region(*_gcs_args())
  print(f"process {jax.process_index()}/{jax.process_count()}, {jax.device_count()} devices "
        f"({jax.local_device_count()} local)", flush=True)
  entry.main(argv)


if __name__ == "__main__":
  app.run(main)
