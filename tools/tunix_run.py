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

  if os.environ.get("GRPO_ENTROPY_METRIC", "0") != "1":
    # Tunix's GRPO loss always asks for per-token entropy, only to log it (aux["entropy"]; it is not in the loss).
    # That is a second full-vocabulary fp32 pass over every token inside the differentiated step: about 2x the
    # training-step memory and much of its time (PPO's loss skips it without an entropy bonus). Skip it; the
    # logged entropy reads 0. GRPO_ENTROPY_METRIC=1 restores it.
    import jax.numpy as _jnp_e  # pylint: disable=g-import-not-at-top
    from tunix.rl import algo_core as _algo_core_e  # pylint: disable=g-import-not-at-top
    from tunix.rl import common as _common_e  # pylint: disable=g-import-not-at-top

    class _NoEntropyCommon:

      def __getattr__(self, name):
        return getattr(_common_e, name)

      @staticmethod
      def compute_per_token_logps(*args, **kwargs):
        if kwargs.get("return_entropy") and not kwargs.get("stop_gradient", True):
          kwargs["return_entropy"] = False
          logps = _common_e.compute_per_token_logps(*args, **kwargs)
          return logps, _jnp_e.zeros_like(logps)
        return _common_e.compute_per_token_logps(*args, **kwargs)

    _algo_core_e.common = _NoEntropyCommon()

if os.environ.get("OVERLONG_FILTER") == "1" and os.environ.get("TUNIX_ALGO", "grpo") == "grpo":
  # DAPO overlong filtering: a rollout cut off at the completion cap (no EOS) is excluded from the policy loss
  # (advantage 0) instead of counting as a failure. With group advantages and a per-sequence token-sum loss, every
  # truncated failure pushes down thousands of reasoning tokens, which steadily shortens the reasoning until the
  # policy answers before it has found a solution (seen on Countdown: GRPO accuracy falls once truncation drops).
  import numpy as _np_ol  # pylint: disable=g-import-not-at-top
  import dataclasses as _dc_ol  # pylint: disable=g-import-not-at-top
  from tunix.rl import rl_cluster as _rc_ol  # pylint: disable=g-import-not-at-top
  from tunix.rl.grpo import grpo_learner as _gl_ol  # pylint: disable=g-import-not-at-top
  _orig_gen_adv_ol = _gl_ol.GRPOLearner._generate_and_compute_advantage  # pylint: disable=protected-access

  def _gen_adv_overlong(self, training_input, mode=_rc_ol.Mode.TRAIN):
    example = _orig_gen_adv_ol(self, training_input, mode)
    if mode != _rc_ol.Mode.TRAIN:
      return example
    truncated = ~(_np_ol.asarray(example.completion_ids) == self.rl_engine.rollout.eos_id()).any(axis=1)
    advantages = _np_ol.asarray(example.advantages).copy()
    advantages[truncated] = 0.0
    self.rl_engine.buffer_metrics({"overlong/frac_truncated": (float(truncated.mean()), _np_ol.mean)}, mode=mode)
    return _dc_ol.replace(example, advantages=jax.numpy.asarray(advantages))

  _gl_ol.GRPOLearner._generate_and_compute_advantage = _gen_adv_overlong  # pylint: disable=protected-access

if os.environ.get("HER") == "1":
  # Goal-conditioned GRPO with hindsight relabeling on Countdown (tools/hindsight.py): each group also trains on
  # copies relabeled to the values its wrong answers reached, importance-corrected by the TIS weights.
  if os.environ.get("TUNIX_ALGO", "grpo") != "grpo":
    raise ValueError("HER=1 needs TUNIX_ALGO=grpo")
  import hindsight  # pylint: disable=g-import-not-at-top
  hindsight.install()

if os.environ.get("LEN_SCHEDULE"):
  # Profiling several completion caps in one job (the model loads once): LEN_SCHEDULE="4096:2,8192:2" runs 2 RL
  # steps at a 4096-token cap, then 2 at 8192, ... The launcher sets MAX_TOKENS to the largest cap (vLLM
  # max_model_len) and NUM_BATCHES to the total. Learners read the rollout config every step, so changing its
  # max_tokens_to_generate changes both the generation cap and the training padding (one recompile per cap).
  # Each step's cap goes to $LOG_ROOT/$RUN_NAME/lengths.jsonl (process 0), aligned with eff.jsonl steps.
  import json as _json_len  # pylint: disable=g-import-not-at-top
  from tunix.rl import rl_cluster as _rc_len  # pylint: disable=g-import-not-at-top
  _len_steps = []
  for _spec in os.environ["LEN_SCHEDULE"].split(","):
    _cap, _n = _spec.split(":")
    _len_steps += [int(_cap)] * int(_n)
  _orig_generate_len = _rc_len.RLEngine.generate

  def _generate_scheduled(self, *args, **kwargs):
    i = getattr(self, "_len_step", 0)
    cap = _len_steps[min(i, len(_len_steps) - 1)]
    self.get_rollout_config(_rc_len.Mode.TRAIN).max_tokens_to_generate = cap
    self._len_step = i + 1  # pylint: disable=protected-access
    if jax.process_index() == 0 and os.environ.get("LOG_ROOT") and os.environ.get("RUN_NAME"):
      with open(os.path.join(os.environ["LOG_ROOT"], os.environ["RUN_NAME"], "lengths.jsonl"), "a") as f:
        f.write(_json_len.dumps({"step": i, "max_tokens": cap}) + "\n")
    print(f"[tunix_run] step {i}: completion cap {cap}", flush=True)
    return _orig_generate_len(self, *args, **kwargs)

  _rc_len.RLEngine.generate = _generate_scheduled

if int(os.environ.get("VLLM_MAX_LEN") or 0) > 0:
  # Tunix sizes vLLM's max_model_len as prompt cap + completion cap + 256. Raise it so in-run evaluation can
  # generate longer than training rollouts (rl_eval per-set caps); the KV pool size does not change.
  import dataclasses as _dc_len  # pylint: disable=g-import-not-at-top
  from tunix.cli import base_rl_pipeline as _brp  # pylint: disable=g-import-not-at-top
  _orig_create_rollout_config = _brp.BasePipeline.create_rollout_config

  def _create_rollout_config(self, *args, **kwargs):
    cfg = _orig_create_rollout_config(self, *args, **kwargs)
    want = int(os.environ["VLLM_MAX_LEN"])
    if getattr(cfg, "kv_cache_size", 0) < want:
      cfg = _dc_len.replace(cfg, kv_cache_size=want)
    return cfg

  _brp.BasePipeline.create_rollout_config = _create_rollout_config

if os.environ.get("TIS") == "1":
  # Rollout-vs-trainer mismatch: vLLM samples with its own kernels / bf16 merged weights, so pi_vllm != pi_trainer
  # on the same tokens. Truncated importance sampling re-weights each token's policy-gradient term by
  # w = min(pi_trainer / pi_vllm, TIS_CAP) (detached). GRPO uses Tunix's built-in version (grpo_config.sampler_is
  # "token", set by the launcher); for PPO the per-token GAE advantages are scaled by w, which equals scaling the
  # clipped per-token loss for w >= 0. Mismatch statistics go to $LOG_ROOT/$RUN_NAME/mismatch.jsonl (process 0).
  import json as _json  # pylint: disable=g-import-not-at-top
  import numpy as _np  # pylint: disable=g-import-not-at-top
  from jax.experimental import multihost_utils as _mhu  # pylint: disable=g-import-not-at-top
  from tunix.rl import rl_cluster as _rc_tis  # pylint: disable=g-import-not-at-top
  _tis_cap = float(os.environ.get("TIS_CAP", "2.0"))

  def _mismatch_log(rec):
    if jax.process_index() == 0 and os.environ.get("LOG_ROOT") and os.environ.get("RUN_NAME"):
      with open(os.path.join(os.environ["LOG_ROOT"], os.environ["RUN_NAME"], "mismatch.jsonl"), "a") as f:
        f.write(_json.dumps(rec) + "\n")

  _orig_buffer_metrics = _rc_tis.RLEngine.buffer_metrics

  def _buffer_metrics(self, metrics, *args, **kwargs):  # GRPO: copy Tunix's sampler/trainer metrics to the jsonl
    sel = {k: float(v[0]) for k, v in metrics.items() if k.startswith(("sampler_trainer/", "sampler_is/"))}
    if sel:
      _mismatch_log(sel)
    return _orig_buffer_metrics(self, metrics, *args, **kwargs)

  _rc_tis.RLEngine.buffer_metrics = _buffer_metrics

  if os.environ.get("TUNIX_ALGO", "grpo") == "ppo":
    _orig_generate_tis = _rc_tis.RLEngine.generate

    def _generate_keep_logprobs(self, *args, **kwargs):
      out = _orig_generate_tis(self, *args, **kwargs)
      self._tis_rollout_logprobs = out.logprobs  # pylint: disable=protected-access
      return out

    _rc_tis.RLEngine.generate = _generate_keep_logprobs

    def _host(x):
      if isinstance(x, jax.Array) and not x.is_fully_addressable:
        return _np.asarray(_mhu.process_allgather(x, tiled=True))
      return _np.asarray(x)

    _orig_gen_adv_tis = ppo_learner.PPOLearner._generate_and_compute_advantage  # pylint: disable=protected-access

    def _gen_adv_tis(self, *args, **kwargs):
      ex = _orig_gen_adv_tis(self, *args, **kwargs)
      roll_lp = getattr(self.rl_engine, "_tis_rollout_logprobs", None)
      if roll_lp is None or ex.old_per_token_logps is None:
        return ex
      old = _host(ex.old_per_token_logps).astype(_np.float32)  # trainer logps (pre-update = current policy)
      mask = _host(ex.completion_mask).astype(_np.float32)
      roll = _np.zeros_like(old)
      for i, x in enumerate(roll_lp):
        n = min(len(x), old.shape[1])
        roll[i, :n] = _np.asarray(x[:n], _np.float32)
      log_ratio = _np.clip(old - roll, -20.0, 20.0)
      ratio = _np.exp(log_ratio)
      w = _np.minimum(ratio, _tis_cap) * mask
      adv = _host(ex.advantages).astype(_np.float32)
      denom = max(float(mask.sum()), 1.0)
      _mismatch_log({"sampler_trainer/logp_diff_mean": float((_np.abs(log_ratio) * mask).sum() / denom),
                     "sampler_trainer/logp_diff_max": float((_np.abs(log_ratio) * mask).max()),
                     "sampler_is/weight_mean": float(w.sum() / denom),
                     "sampler_is/frac_clipped": float(((ratio > _tis_cap) * mask).sum() / denom)})
      return _dc_tis.replace(ex, advantages=jax.numpy.asarray(adv * w if adv.ndim == 2 else adv[:, None] * w))

    import dataclasses as _dc_tis  # pylint: disable=g-import-not-at-top
    ppo_learner.PPOLearner._generate_and_compute_advantage = _gen_adv_tis  # pylint: disable=protected-access

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
