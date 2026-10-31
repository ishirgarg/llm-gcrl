"""Countdown reward for Tunix (every public function in this file is loaded as a reward function).

reward = 1 if the <answer> expression uses exactly the given numbers and evaluates to the target, else 0.
Per-step stats over the real (not hindsight-relabeled) rollouts go to $LOG_ROOT/$RUN_NAME/rewards.jsonl:
acc, answer_rate (has an <answer> tag) and valid_rate (a valid expression over the numbers, right or wrong).
"""
import json
import os
import time

import countdown


def _log(goals, completions, scores, relabeled):
  if os.environ.get("WORKER") not in ("w-0", None) or not os.environ.get("RUN_NAME"):
    return
  real = [i for i in range(len(scores)) if not relabeled[i]]
  if not real:
    return
  n = len(real)
  rec = {"time": round(time.time(), 1), "n": n, "acc": sum(scores[i] for i in real) / n,
         "answer_rate": sum(countdown.formatted(completions[i]) for i in real) / n,
         "valid_rate": sum(countdown.achieved(goals[i], completions[i]) is not None for i in real) / n}
  try:
    with open(os.path.join(os.environ["LOG_ROOT"], os.environ["RUN_NAME"], "rewards.jsonl"), "a") as f:
      f.write(json.dumps(rec) + "\n")
  except OSError:
    pass


def countdown_reward(prompts, completions, numbers, target, relabeled=None, **kwargs):
  """Tunix reward fn: per-row goal fields come from the dataset (and from tools/hindsight.py for relabeled rows)."""
  del prompts, kwargs
  goals = [countdown.Goal.decode(n, t) for n, t in zip(numbers, target)]
  scores = [countdown.reward(g, c) for g, c in zip(goals, completions)]
  _log(goals, completions, scores, relabeled if relabeled is not None else [False] * len(scores))
  return scores
