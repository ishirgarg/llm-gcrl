"""One definition of the efficiency metrics, shared by every framework so runs are comparable.

All metrics are per RL step and logged under `eff/`:
  step_time_s, rollout_time_s, train_time_s       wall-clock seconds
  gen_tokens, prompt_tokens, samples              rollout volume (gen = sampled completion tokens)
  rollout_tps[_per_chip]   gen_tokens / rollout_time_s              (sampling throughput)
  train_tps[_per_chip]     (prompt + gen tokens) / train_time_s     (trainer throughput)
  e2e_tps[_per_chip]       gen_tokens / step_time_s                 (headline TPS)
  samples_per_s            samples / step_time_s
  train_mfu_6n             6 * N * (prompt + gen) / (train_time_s * chips * peak_flops)
                           (6N convention, counts only the policy update; LoRA, reference/critic passes and
                           remat are not modelled, so compare it across runs rather than reading it absolutely)
"""

import os

QWEN3_4B_PARAMS = 4.02e9
# Parameters counted by the 6N MFU convention (MoE: active parameters per token).
MODEL_PARAMS = {
    "Qwen/Qwen3-1.7B": 2.03e9,
    "Qwen/Qwen3-4B": QWEN3_4B_PARAMS,
    "Qwen/Qwen3-4B-Instruct-2507": QWEN3_4B_PARAMS,
    "Qwen/Qwen3-8B": 8.19e9,
    "Qwen/Qwen3-14B": 14.77e9,
    "Qwen/Qwen3-30B-A3B": 3.35e9,
}
V5E_PEAK_BF16_FLOPS = 197e12
N_CHIPS = 16


def derive(step_time_s, rollout_time_s, train_time_s, gen_tokens, prompt_tokens, samples,
           n_chips=N_CHIPS, n_params=None, peak_flops=V5E_PEAK_BF16_FLOPS):
  if n_params is None:  # launchers export MODEL (HF id)
    model = os.environ.get("MODEL", "Qwen/Qwen3-4B")
    n_params = MODEL_PARAMS[model[:-len("-Base")] if model.endswith("-Base") else model]
  def div(a, b):
    return a / b if a is not None and b else None

  train_tokens = (gen_tokens or 0) + (prompt_tokens or 0)
  m = {
      "step_time_s": step_time_s,
      "rollout_time_s": rollout_time_s,
      "train_time_s": train_time_s,
      "gen_tokens": gen_tokens,
      "prompt_tokens": prompt_tokens,
      "samples": samples,
      "rollout_tps": div(gen_tokens, rollout_time_s),
      "train_tps": div(train_tokens, train_time_s),
      "e2e_tps": div(gen_tokens, step_time_s),
      "samples_per_s": div(samples, step_time_s),
      "train_mfu_6n": div(6 * n_params * train_tokens, (train_time_s or 0) * n_chips * peak_flops),
  }
  for k in ("rollout_tps", "train_tps", "e2e_tps"):
    m[f"{k}_per_chip"] = div(m[k], n_chips)
  return {f"eff/{k}": v for k, v in m.items() if v is not None}
