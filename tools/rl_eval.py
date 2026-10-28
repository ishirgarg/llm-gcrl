"""Evaluation inside a Tunix RL run, sampled through the run's own vLLM rollout engine (= current policy weights).

  EVAL_SETS="math500:4,amc23:8,aime24:8"   sets (tools/math_data.py) and samples per problem
  EVAL_EVERY=25                            before training, every N policy updates, and after the last one
  EVAL_TEMPERATURE / EVAL_TOP_P            sampling for evaluation (default: the training rollout's)

Per set: accuracy (mean over all samples, = avg@k), pass@k, \\boxed{} rate and mean completion length, appended
to $LOG_ROOT/$RUN_NAME/eval.jsonl by process 0 and buffered to the metrics logger as eval/<set>/<metric>.
Evaluation time is left out of the eff/* step times. install() wraps RLEngine.generate's raw version before
tunix_eff instruments it, so evaluation generations are not counted as rollouts.
"""
import dataclasses
import json
import os
import time

import jax
import math_data
import math_reward
import numpy as np
from tunix.rl import rl_cluster

_raw_generate = None
_sets = None
_done_steps = set()


def _load_sets():
  global _sets
  if _sets is None:
    _sets = []
    for spec in os.environ["EVAL_SETS"].split(","):
      name, k = (spec.split(":") + ["1"])[:2]
      _sets.append((name, int(k), math_data.eval_set(name)))
  return _sets


def _eval_rollout_config(engine):
  cfg = engine.cluster_config.rollout_config
  overrides = {}
  if os.environ.get("EVAL_TEMPERATURE"):
    overrides["temperature"] = float(os.environ["EVAL_TEMPERATURE"])
  if os.environ.get("EVAL_TOP_P"):
    overrides["top_p"] = float(os.environ["EVAL_TOP_P"])
  return cfg, (dataclasses.replace(cfg, **overrides) if overrides and not isinstance(cfg, dict) else cfg)


def evaluate(engine, step):
  if step in _done_steps:
    return
  _done_steps.add(step)
  t_all = time.perf_counter()
  orig_cfg, eval_cfg = _eval_rollout_config(engine)
  object.__setattr__(engine.cluster_config, "rollout_config", eval_cfg)
  try:
    for name, k, items in _load_sets():
      t0 = time.perf_counter()
      prompts = [p for p, _ in items for _ in range(k)]
      out = _raw_generate(engine, prompts, micro_batch_size=len(prompts))
      scores = np.array([math_reward._score(c, a) for c, (_, a) in  # pylint: disable=protected-access
                         zip(out.text, [it for it in items for _ in range(k)])]).reshape(len(items), k)
      correct = scores == 1.0
      m = {
          "acc": float(correct.mean()),
          f"pass@{k}": float(correct.any(axis=1).mean()),
          "boxed_rate": float((scores != -math_reward.FORMAT_COEF).mean()),
          "mean_len": float(np.mean([len(t) for t in out.tokens])),
          "max_len_frac": float(np.mean([len(t) >= eval_cfg.max_tokens_to_generate for t in out.tokens])),
      }
      dt = time.perf_counter() - t0
      if jax.process_index() == 0:
        path = os.path.join(os.environ["LOG_ROOT"], os.environ["RUN_NAME"], "eval.jsonl")
        with open(path, "a") as f:
          f.write(json.dumps({"step": step, "set": name, "n_problems": len(items), "samples": k, **m,
                              "time_s": round(dt, 1)}) + "\n")
      print(f"[eval] step {step} {name} (n={len(items)}x{k}): " + ", ".join(f"{a}={b:.4f}" for a, b in m.items())
            + f" ({dt:.0f}s)", flush=True)
      try:
        engine.buffer_metrics({f"eval/{name}/{a}": (b, np.mean) for a, b in m.items()})
      except Exception:  # pylint: disable=broad-except
        pass
  finally:
    object.__setattr__(engine.cluster_config, "rollout_config", orig_cfg)
  import tunix_eff  # pylint: disable=g-import-not-at-top
  tunix_eff.exclude(time.perf_counter() - t_all)


def install():
  """Call before tunix_eff.install(): keeps the uninstrumented generate for evaluation."""
  global _raw_generate
  _raw_generate = rl_cluster.RLEngine.generate


def install_hooks():
  """Call after tunix_eff.install(): evaluation runs outside the instrumented sync and before / after training."""
  every = int(os.environ.get("EVAL_EVERY", "25"))
  orig_sync = rl_cluster.RLEngine.sync_weights

  def sync_then_eval(self):
    out = orig_sync(self)
    self._eval_syncs = getattr(self, "_eval_syncs", 0) + 1  # pylint: disable=protected-access
    n = self._eval_syncs  # pylint: disable=protected-access
    max_steps = getattr(self.cluster_config.training_config, "max_steps", None)
    if n % every == 0 or n == max_steps:
      evaluate(self, n)
    return out

  rl_cluster.RLEngine.sync_weights = sync_then_eval

  from tunix.rl import rl_learner  # pylint: disable=g-import-not-at-top
  learners = [rl_learner.RLLearner]
  try:
    from tunix.rl.grpo import grpo_learner  # pylint: disable=g-import-not-at-top
    learners.append(grpo_learner.GRPOLearner)
  except ImportError:
    pass
  try:
    from tunix.rl.ppo import ppo_learner  # pylint: disable=g-import-not-at-top
    learners.append(ppo_learner.PPOLearner)
  except ImportError:
    pass
  for cls in learners:
    if "train" not in cls.__dict__:
      continue
    orig_train = cls.__dict__["train"]

    def train_with_eval(self, *args, _orig=orig_train, **kwargs):
      if getattr(self, "_eval_in_train", False):  # subclass train delegating to the base: evaluate once
        return _orig(self, *args, **kwargs)
      self._eval_in_train = True
      evaluate(self.rl_engine, 0)  # base policy: vLLM still holds the initial weights
      out = _orig(self, *args, **kwargs)
      steps = getattr(self.rl_engine, "_eval_syncs", 0)
      if steps and steps not in _done_steps:
        evaluate(self.rl_engine, steps)
      return out

    cls.train = train_with_eval
