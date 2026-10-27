"""Turn a KV probe run into measured numbers (runs on the VM; reads small NFS logs only).

For each probe window it takes the per-chip peak HBM from hw-*.csv (max over all chips/workers) and the idle
level just before the window. The KV cost per token is the slope of peak HBM against allocated tokens
(num_samples x allocated cache length) between lengths, which cancels weights and fixed buffers. Throughput uses the
second (post-compile) repeat of each length.

Usage: python kv_report.py <run> [--n-chips 16]
"""
import argparse
import csv
import glob
import json
import os

LOG_ROOT = os.environ["LOG_ROOT"]  # from scripts/vm_env.sh


def skyrl_round_up_seq_len(seq_len):
  """Copy of skyrl.tx.utils.models.round_up_seq_len: SkyRL's dense KV cache is allocated at this length."""
  if seq_len <= 32:
    return 32
  msb = seq_len.bit_length() - 1
  result = seq_len & ((1 << msb) | (1 << (msb - 1)))
  return result + (1 << (msb - 1)) if result < seq_len else result


def main():
  p = argparse.ArgumentParser()
  p.add_argument("run")
  p.add_argument("--n-chips", type=int, default=16)
  p.add_argument("--skyrl-rounding", action="store_true", help="dense cache length = round_up(prompt + max_tokens)")
  args = p.parse_args()
  d = os.path.join(LOG_ROOT, args.run)

  samples = []  # (time, hbm_bytes)
  for path in glob.glob(os.path.join(d, "hw-*.csv")):
    with open(path) as f:
      samples += [(float(r["time"]), int(r["hbm_used_bytes"])) for r in csv.DictReader(f)]
  probes = [json.loads(l) for l in open(os.path.join(d, "kv_probe.jsonl"))]

  rows = []
  for pr in probes:
    window = [h for t, h in samples if pr["t0"] <= t <= pr["t1"] + 2]
    before = [h for t, h in samples if pr["t0"] - 8 <= t < pr["t0"]]
    if not window:
      continue
    rows.append({**pr, "peak": max(window), "idle": max(before) if before else None,
                 "alloc_tokens": pr["num_samples"] * pr["max_tokens"]})

  gib = 2**30
  print("| max_tokens | repeat | gen tokens | time s | gen tok/s | gen tok/s/chip | peak HBM GiB/chip | idle GiB/chip |")
  print("|---|---|---|---|---|---|---|---|")
  for r in rows:
    dt = r["t1"] - r["t0"]
    print(f"| {r['max_tokens']} | {r['repeat']} | {r['gen_tokens']} | {dt:.1f} | {r['gen_tokens']/dt:,.0f} | "
          f"{r['gen_tokens']/dt/args.n_chips:,.0f} | {r['peak']/gib:.2f} | {(r['idle'] or 0)/gib:.2f} |")

  def cache_len(r):
    if args.skyrl_rounding:  # SkyRL pads prompts to a bucket, then rounds prompt + max_tokens again
      return skyrl_round_up_seq_len(skyrl_round_up_seq_len(r["prompt_tokens"]) + r["max_tokens"])
    return r["prompt_tokens"] + r["max_tokens"]

  by_len = {}
  for r in rows:
    by_len[cache_len(r)] = max(by_len.get(cache_len(r), 0), r["peak"])
  lens = sorted(by_len)
  n = rows[0]["num_samples"]
  print("\ncache lengths per sequence (allocated):", lens)
  for a, b in zip(lens, lens[1:]):
    per_chip = (by_len[b] - by_len[a]) / (n * (b - a))
    total = per_chip * args.n_chips
    print(f"KV slope {a}->{b}: {total/1024:.1f} KiB/token total ({per_chip:.0f} B/token/chip) "
          f"= {total*1000/gib:.3f} GiB per 1k tokens (one sequence)")


if __name__ == "__main__":
  main()
