"""Adds the shared eff/* efficiency metrics to Tunix-based RL (Tunix PPO and MaxText RL both use RLEngine).

Patches RLEngine to time the rollout phase (`generate`), the trainer phase (logprob / value passes, critic and
actor updates) and the trainer->rollout weight sync (`sync_weights`, as eff/sync_time_s), and to count prompt / completion tokens from rollout outputs.
A step ends when the next rollout starts (update_actor runs once per micro-batch); at that point it:
  - buffers eff/* into Tunix's metrics logger (-> wandb / tensorboard; Tunix flushes them with the next
    step's metrics, so wandb shows them one step late), and
  - appends the exact per-step record to $LOG_ROOT/$RUN_NAME/eff.jsonl on process 0
    (eff-w-N.jsonl per host in single-host mode).
"""
import atexit
import json
import os
import time

import effmetrics
import jax
import numpy as np
from tunix.rl import rl_cluster

_TRAIN_PHASES = ("get_ref_per_token_logps", "get_old_per_token_logps", "get_values", "update_critic", "update_actor")
_SYNC_PHASES = ("sync_weights",)  # trainer -> rollout weight transfer, reported separately as eff/sync_time_s
_CRITIC_PHASES = ("get_values", "update_critic")  # PPO critic; also counted in train_time (eff/critic_time_s)


class _Acc:

  def __init__(self):
    self.reset(time.perf_counter())

  def reset(self, now):
    self.step_start = now
    self.rollout_s = 0.0
    self.train_s = 0.0
    self.sync_s = 0.0
    self.critic_s = 0.0
    self.gen_tokens = 0
    self.prompt_tokens = 0
    self.samples = 0


_acc = _Acc()
_step = 0


def _jsonl_path():
  root, run = os.environ.get("LOG_ROOT"), os.environ.get("RUN_NAME")
  if not (root and run):
    return None
  # In single-host mode every host is an independent "process 0": one file per host.
  name = f"eff-{os.environ['WORKER']}.jsonl" if os.environ.get("SINGLE_HOST") == "1" else "eff.jsonl"
  return os.path.join(root, run, name)


def _count(engine, out):
  _acc.samples += len(out.tokens)
  _acc.gen_tokens += int(sum(len(t) for t in out.tokens))
  prompts = np.asarray(out.left_padded_prompt_tokens)
  pad = None
  tok = getattr(engine, "tokenizer", None)
  for attr in ("pad_id", "pad_token_id"):
    v = getattr(tok, attr, None)
    pad = v() if callable(v) else v
    if pad is not None:
      break
  _acc.prompt_tokens += int(np.count_nonzero(prompts != (0 if pad is None else pad)))


_engine = None


def _wrap(name, phase):
  orig = getattr(rl_cluster.RLEngine, name)

  def wrapped(self, *args, **kwargs):
    global _engine
    _engine = self
    if phase == "rollout" and _acc.samples and _acc.train_s:
      # A new rollout starts the next RL step. (update_actor runs once per micro-batch, so it can't mark steps.)
      _end_step(self)
    t0 = time.perf_counter()
    out = orig(self, *args, **kwargs)
    dt = time.perf_counter() - t0
    if phase == "rollout":
      _acc.rollout_s += dt
      _count(self, out)
    elif phase == "sync":
      _acc.sync_s += dt
    else:
      _acc.train_s += dt
      if name in _CRITIC_PHASES:
        _acc.critic_s += dt
    return out

  setattr(rl_cluster.RLEngine, name, wrapped)


def _flush_last_step():
  if _engine is not None and _acc.samples and _acc.train_s:
    _end_step(_engine, log_to_engine=False)


def _end_step(engine, log_to_engine=True):
  global _step
  now = time.perf_counter()
  m = effmetrics.derive(
      step_time_s=now - _acc.step_start,
      rollout_time_s=_acc.rollout_s,
      train_time_s=_acc.train_s,
      gen_tokens=_acc.gen_tokens,
      prompt_tokens=_acc.prompt_tokens,
      samples=_acc.samples,
      n_chips=jax.device_count(),
  )
  m["eff/sync_time_s"] = _acc.sync_s
  m["eff/critic_time_s"] = _acc.critic_s
  if log_to_engine:
    engine.buffer_metrics({k: (v, np.mean) for k, v in m.items()})
  path = _jsonl_path()
  if path and jax.process_index() == 0:
    with open(path, "a") as f:
      f.write(json.dumps({"step": _step, **m}) + "\n")
  print(f"[eff] step {_step}: " + ", ".join(f"{k[4:]}={v:.4g}" for k, v in m.items()), flush=True)
  _step += 1
  _acc.reset(time.perf_counter())


def install():
  atexit.register(_flush_last_step)  # the last step has no following rollout to close it
  _wrap("generate", "rollout")
  for name in _TRAIN_PHASES:
    if hasattr(rl_cluster.RLEngine, name):
      _wrap(name, "train")
  for name in _SYNC_PHASES:
    if hasattr(rl_cluster.RLEngine, name):
      _wrap(name, "sync")
