"""Math RL reward shared by the Tunix and MaxText runs, matching the SkyRL / tinker-cookbook MathEnv:

  reward = correct + 0.1 * (has_boxed - 1)

`correct` uses the cookbook's own sympy grader (math_grading.py, loaded from the NFS checkout) on the last
\\boxed{...}. One difference from the cookbook: the cookbook also withholds the format bonus from truncated
samples; these frameworks don't pass truncation info to reward functions, so only \\boxed presence counts.
"""
import concurrent.futures
import importlib.util
import json
import os
import time

_GRADING = os.path.join(os.environ["NFS_ROOT"], "third_party/tinker-cookbook/tinker_cookbook/recipes/math_rl/math_grading.py")
_spec = importlib.util.spec_from_file_location("cookbook_math_grading", _GRADING)
grading = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(grading)

_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)
FORMAT_COEF = 0.1
TIMEOUT_S = 1.0


def _score(completion: str, answer: str) -> float:
  try:
    given = grading.extract_boxed(completion)
  except ValueError:
    return -FORMAT_COEF
  try:
    correct = _pool.submit(grading.grade_answer, given, str(answer)).result(timeout=TIMEOUT_S)
  except Exception:  # timeout or sympy failure counts as wrong
    correct = False
  return float(correct)


def _log(scores):
  """Per-call reward stats (= per training step) to $LOG_ROOT/$RUN_NAME/rewards.jsonl, written by worker 0 only."""
  if os.environ.get("WORKER") not in ("w-0", None) or not os.environ.get("RUN_NAME") or not scores:
    return
  path = os.path.join(os.environ["LOG_ROOT"], os.environ["RUN_NAME"], "rewards.jsonl")
  rec = {"time": round(time.time(), 1), "n": len(scores), "mean_reward": sum(scores) / len(scores),
         "acc": sum(s == 1.0 for s in scores) / len(scores),
         "boxed_rate": sum(s != -FORMAT_COEF for s in scores) / len(scores)}
  try:
    with open(path, "a") as f:
      f.write(json.dumps(rec) + "\n")
  except OSError:
    pass


def math_reward(prompts, completions, answer=None, **kwargs):
  """Tunix-style reward fn: (prompts, completions, answer, **kw) -> list[float]."""
  del prompts
  if answer is None:
    answer = kwargs["answers"]
  scores = [_score(c, a) for c, a in zip(completions, answer)]
  _log(scores)
  return scores
