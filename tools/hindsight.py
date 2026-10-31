"""Hindsight relabeling for goal-conditioned GRPO on Countdown (tools/countdown.py), installed by tools/tunix_run.py
when HER=1.

Every prompt's group of G rollouts is sampled for its goal g = (numbers, target) as usual. Afterwards each group
gets HER_GOALS extra copies, each relabeled to a goal g' = (numbers, v) where v is a value that one of the group's
wrong but valid answers actually reached (distinct values, in the training target range). A relabeled copy holds
the same G completions under the prompt for g', re-scored against g': the completions that reach v get reward 1,
the rest 0. Its group advantage is computed like any other group's (Dr. GRPO: reward minus the group mean), so a
group in which every rollout failed, which gives GRPO no gradient, still teaches the policy which expressions reach
which values. No extra rollouts are sampled: the env-step budget is the same as plain GRPO.

Off-policy correction: the completions were sampled from pi(. | g), not pi(. | g'). The trainer's log-prob pass runs
on the relabeled prompt, so the truncated importance weight GRPO already applies for the vLLM-vs-trainer mismatch
(grpo_config.sampler_is="token", TIS=1) becomes w_t = min(pi_train(y_t | g', y_<t) / pi_vllm(y_t | g, y_<t), cap)
for relabeled rows: tokens that only made sense for the original target (e.g. "we need 24") are down-weighted.

Loss = GRPO loss over real and relabeled rows, relabeled advantages scaled by HER_COEF. A relabeled copy with no
usable value (every answer right, or none valid) gets zero advantages (it keeps the batch shape static).

  HER_GOALS=1    relabeled copies per group (the training batch grows (1 + HER_GOALS)x; rollouts do not)
  HER_COEF=1.0   weight of the relabeled rows' advantages

Per-step stats go to $LOG_ROOT/$RUN_NAME/hindsight.jsonl (process 0).
"""
import dataclasses
import hashlib
import json
import os
import random

import countdown
import jax
from jax.experimental import multihost_utils
import jax.numpy as jnp
import numpy as np
from tunix.rl import rl_cluster
from tunix.rl import rl_learner
from tunix.rl.grpo import grpo_learner

GOALS = int(os.environ.get("HER_GOALS", "1"))
COEF = float(os.environ.get("HER_COEF", "1.0"))


def relabel_goals(goal, completions, k, rng):
  """Up to k distinct in-range values reached by valid wrong answers, as new goals."""
  lo, hi = countdown.target_range()
  reached = {countdown.achieved(goal, c) for c in completions}
  values = sorted(v for v in reached if v is not None and v != goal.target and lo <= v <= hi)
  rng.shuffle(values)
  return [countdown.Goal(goal.numbers, v) for v in values[:k]]


def _encode_prompt(engine, text, width):
  """Left-padded prompt tokens, tokenized the way the rollout sampler does (BOS only if the tokenizer has one)."""
  tok = engine.tokenizer
  ids = ([tok.bos_id()] if tok.bos_id() else []) + list(tok.encode(text))
  if len(ids) > width:
    raise ValueError(f"relabeled prompt has {len(ids)} tokens > prompt width {width}; raise MAX_PROMPT")
  return np.array([engine.rollout.pad_id()] * (width - len(ids)) + ids, dtype=np.int32)


def _augment(learner, training_input, out):
  """Append HER_GOALS relabeled copies after every group: rows [g_0 | g'_0,1..K | g_1 | g'_1,1..K | ...]."""
  engine = learner.rl_engine
  g = learner.algo_config.num_generations
  n_groups = len(out.text) // g
  prompts = list(training_input["prompts"])
  width = out.left_padded_prompt_tokens.shape[1]
  if not np.array_equal(_encode_prompt(engine, prompts[0], width), out.left_padded_prompt_tokens[0]):
    raise RuntimeError("hindsight: re-tokenized prompt differs from the sampler's tokens")

  src, new_prompt, new_target, prompt_tokens, kind = [], [], [], [], []  # kind: 0 real, 1 relabeled, 2 unused
  for p in range(n_groups):
    rows = list(range(p * g, (p + 1) * g))
    goal = countdown.Goal.decode(training_input["numbers"][rows[0]], training_input["target"][rows[0]])
    texts = [out.text[i] for i in rows]
    seed = hashlib.sha256((prompts[rows[0]] + "\0".join(texts)).encode()).hexdigest()  # same on every host
    new_goals = relabel_goals(goal, texts, GOALS, random.Random(seed))
    src += rows
    new_prompt += [prompts[i] for i in rows]
    new_target += [goal.target] * g
    prompt_tokens += [out.left_padded_prompt_tokens[i] for i in rows]
    kind += [0] * g
    for j in range(GOALS):
      relabeled = new_goals[j] if j < len(new_goals) else goal  # unused copies keep the real goal, masked below
      text = countdown.prompt(relabeled) if j < len(new_goals) else prompts[rows[0]]
      tokens = _encode_prompt(engine, text, width)
      src += rows
      new_prompt += [text] * g
      new_target += [relabeled.target] * g
      prompt_tokens += [tokens] * g
      kind += [1 if j < len(new_goals) else 2] * g

  n = len(out.text)
  for key, value in list(training_input.items()):  # every per-row field follows its source row
    if isinstance(value, (list, np.ndarray)) and len(value) == n:
      training_input[key] = [value[i] for i in src] if isinstance(value, list) else np.asarray(value)[src]
  training_input["prompts"] = new_prompt
  training_input["target"] = np.array([str(t) for t in new_target])
  training_input["relabeled"] = np.array([k != 0 for k in kind])
  learner._her_kind = np.array(kind)  # pylint: disable=protected-access
  return dataclasses.replace(
      out, text=[out.text[i] for i in src], tokens=[out.tokens[i] for i in src],
      logprobs=None if out.logprobs is None else [out.logprobs[i] for i in src],
      left_padded_prompt_tokens=np.stack(prompt_tokens), logits=None, routed_experts=None)


def _host(x):
  """Numpy copy of a (possibly multi-host sharded) array; a collective, so every process must call it."""
  if isinstance(x, jax.Array) and not x.is_fully_addressable:
    return np.asarray(multihost_utils.process_allgather(x, tiled=True))
  return np.asarray(x)


def _log_stats(learner, example, training_input):
  weights = None if example.sampler_is_weights is None else _host(example.sampler_is_weights)
  mask = _host(example.completion_mask).astype(np.float32)
  if jax.process_index() != 0 or not os.environ.get("RUN_NAME"):
    return
  g = learner.algo_config.num_generations
  kind = learner._her_kind  # pylint: disable=protected-access
  wins = np.array([countdown.reward(countdown.Goal.decode(nu, t), c) for nu, t, c in zip(
      training_input["numbers"], training_input["target"], learner._her_text)])  # pylint: disable=protected-access
  per_group = wins.reshape(-1, g).sum(axis=1)
  group_kind = kind.reshape(-1, g)[:, 0]
  real, relab = per_group[group_kind == 0], per_group[group_kind == 1]
  rec = {"groups": int(len(real)), "real_acc": float(real.mean() / g),
         "real_all_fail": float((real == 0).mean()), "real_all_win": float((real == g).mean()),
         "relabeled_groups": int(len(relab)), "relabeled_per_group": float(len(relab) / max(len(real), 1)),
         "relabeled_wins_mean": float(relab.mean()) if len(relab) else 0.0}
  if weights is not None:
    for name, sel in (("real", kind == 0), ("relabeled", kind == 1)):
      if sel.any():
        rec[f"is_weight_{name}"] = float((weights[sel] * mask[sel]).sum() / max(mask[sel].sum(), 1.0))
  with open(os.path.join(os.environ["LOG_ROOT"], os.environ["RUN_NAME"], "hindsight.jsonl"), "a") as f:
    f.write(json.dumps(rec) + "\n")


def install():
  orig_gen_adv = grpo_learner.GRPOLearner._generate_and_compute_advantage  # pylint: disable=protected-access

  def gen_adv_hindsight(self, training_input, mode=rl_cluster.Mode.TRAIN):
    if mode != rl_cluster.Mode.TRAIN:
      return orig_gen_adv(self, training_input, mode)
    engine = self.rl_engine
    generate = engine.generate

    def generate_and_relabel(*args, **kwargs):
      out = _augment(self, training_input, generate(*args, **kwargs))
      self._her_text = out.text  # pylint: disable=protected-access
      return out

    engine.generate = generate_and_relabel  # instance attribute shadows the (eff-instrumented) class method
    try:
      example = orig_gen_adv(self, training_input, mode)
    finally:
      del engine.generate
    kind = self._her_kind  # pylint: disable=protected-access
    scale = np.where(kind == 0, 1.0, np.where(kind == 1, COEF, 0.0)).astype(np.float32)
    advantages = _host(example.advantages)
    advantages = advantages * (scale if advantages.ndim == 1 else scale[:, None])
    example = dataclasses.replace(example, advantages=jnp.asarray(advantages))
    _log_stats(self, example, training_input)
    return example

  grpo_learner.GRPOLearner._generate_and_compute_advantage = gen_adv_hindsight  # pylint: disable=protected-access

  # The learner splits each processed batch back into training micro-batches of the sizes it sampled; with the
  # relabeled copies every micro-batch holds (1 + HER_GOALS)x the rows (its prompts' real and relabeled groups).
  orig_process = rl_learner.RLLearner._process_accumulated_batches  # pylint: disable=protected-access

  def process_hindsight(self, micro_batches, micro_batch_sizes, mode):
    if isinstance(self, grpo_learner.GRPOLearner) and mode == rl_cluster.Mode.TRAIN:
      micro_batch_sizes = [n * (1 + GOALS) for n in micro_batch_sizes]
    return orig_process(self, micro_batches, micro_batch_sizes, mode)

  rl_learner.RLLearner._process_accumulated_batches = process_hindsight  # pylint: disable=protected-access
