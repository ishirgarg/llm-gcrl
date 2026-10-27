"""Compare efficiency across runs. Runs on the VM; reads only small text logs on NFS.

For each run dir under $LOG_ROOT it reads per-step eff/* metrics (SkyRL: client/metrics.jsonl,
Tunix/MaxText: eff.jsonl) and the hw-*.csv sidecar files, drops the first (compile) step, and prints a
markdown table of medians.

Usage: python tools/eff_report.py <run> [<run> ...]
"""
import csv
import glob
import json
import os
import statistics
import sys

LOG_ROOT = os.environ["LOG_ROOT"]  # from scripts/vm_env.sh
COLS = [
    ("steps", "steps"),
    ("eff/step_time_s", "step s"),
    ("eff/rollout_time_s", "rollout s"),
    ("eff/train_time_s", "train s"),
    ("eff/gen_tokens", "gen tok/step"),
    ("eff/rollout_tps_per_chip", "rollout tok/s/chip"),
    ("eff/train_tps_per_chip", "train tok/s/chip"),
    ("eff/e2e_tps", "e2e gen tok/s"),
    ("eff/samples_per_s", "samples/s"),
    ("eff/train_mfu_6n", "train MFU (6N)"),
    ("duty", "TPU duty %"),
    ("hbm", "peak HBM GiB/chip"),
]


def step_records(run_dir):
  for path in (os.path.join(run_dir, "client", "metrics.jsonl"), os.path.join(run_dir, "eff.jsonl")):
    if os.path.exists(path):
      with open(path) as f:
        return [r for r in map(json.loads, f) if "eff/step_time_s" in r]
  return []


def hw_stats(run_dir):
  duty, peak = [], 0
  for path in glob.glob(os.path.join(run_dir, "hw-*.csv")):
    with open(path) as f:
      for row in csv.DictReader(f):
        duty.append(float(row["duty_cycle_pct"]))
        peak = max(peak, int(row["hbm_used_bytes"]))
  busy = [d for d in duty if d > 0]
  return (statistics.mean(busy) if busy else None), (peak / 2**30 if peak else None)


def summarize(run):
  run_dir = os.path.join(LOG_ROOT, run)
  recs = step_records(run_dir)[1:]  # drop the compile step
  row = {"steps": len(recs)}
  for key, _ in COLS:
    vals = [r[key] for r in recs if r.get(key) is not None] if key.startswith("eff/") else []
    if vals:
      row[key] = statistics.median(vals)
  row["duty"], row["hbm"] = hw_stats(run_dir)
  return row


def fmt(v):
  if v is None:
    return "-"
  if isinstance(v, int):
    return str(v)
  return f"{v:.3g}" if abs(v) < 1000 else f"{v:,.0f}"


def main():
  runs = sys.argv[1:]
  print("| run | " + " | ".join(h for _, h in COLS) + " |")
  print("|---" * (len(COLS) + 1) + "|")
  for run in runs:
    r = summarize(run)
    print(f"| {run} | " + " | ".join(fmt(r.get(k)) for k, _ in COLS) + " |")


if __name__ == "__main__":
  main()
