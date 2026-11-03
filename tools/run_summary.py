"""Compact summary of one Tunix run's logs (stdlib only): per-step eff/* times, medians over steps >= 1 (step 0
includes compilation), peak HBM per chip over all workers, vLLM KV cache size, first OOM / error lines, and the
evaluation results (eval.jsonl).

usage: python3 tools/run_summary.py $LOG_ROOT/<run>
"""
import csv
import glob
import json
import os
import re
import statistics
import sys

KEYS = ("step_time_s", "rollout_time_s", "train_time_s", "critic_time_s", "sync_time_s", "gen_tokens", "e2e_tps")


def main(d):
  steps = []
  if os.path.exists(f"{d}/eff.jsonl"):
    for line in open(f"{d}/eff.jsonl"):
      r = json.loads(line)
      steps.append({k: r.get(f"eff/{k}") for k in KEYS} | {"step": r["step"]})
  for s in steps:
    print("  step", s["step"], {k: round(v, 1) for k, v in s.items() if k != "step" and v is not None})
  if os.path.exists(f"{d}/lengths.jsonl"):  # LEN_SCHEDULE run: last step at each completion cap (first one compiles)
    cap_of = {r["step"]: r["max_tokens"] for r in map(json.loads, open(f"{d}/lengths.jsonl"))}
    by_cap = {}
    for s in steps:
      by_cap.setdefault(cap_of.get(s["step"]), []).append(s)
    for cap, ss in by_cap.items():
      last = ss[-1]
      print(f"  CAP {cap}: steps {len(ss)}, last:", {k: round(v, 1) for k, v in last.items() if k != "step" and v is not None})
    reached = [c for c in cap_of.values()]
    if reached:
      print(f"  CAPS reached: up to {max(reached)} (completed steps per cap above)")
  later = steps[1:] if len(steps) > 1 else steps
  if later:
    med = {k: round(statistics.median([s[k] for s in later if s[k] is not None]), 1)
           for k in KEYS if any(s[k] is not None for s in later)}
    print(f"  MEDIAN steps>=1 (n={len(later)}):", med)
  peak, total = 0, 0
  for f in glob.glob(f"{d}/hw-*.csv"):
    with open(f) as fh:
      for row in csv.DictReader(fh):
        try:
          peak = max(peak, int(row["hbm_used_bytes"]))
          total = int(row["hbm_total_bytes"])
        except (KeyError, ValueError):
          pass
  print(f"  peak HBM {peak / 2**30:.2f} / {total / 2**30:.2f} GiB per chip")
  kv, oom, err = set(), [], []
  for f in sorted(glob.glob(f"{d}/w-*.log")):
    for line in open(f, errors="replace"):
      if "GPU KV cache size" in line:
        m = re.search(r"GPU KV cache size: ([\d,]+) tokens", line)
        kv.add(m.group(1) if m else line.strip()[-60:])
      elif ("RESOURCE_EXHAUSTED" in line or "Attempting to allocate" in line or "exceeds" in line
            and "hbm" in line.lower()) and len(oom) < 3:
        oom.append(f"{os.path.basename(f)}: {line.strip()[:220]}")
      elif re.match(r"^\w*(Error|Exception)\b", line) and len(err) < 2:
        err.append(f"{os.path.basename(f)}: {line.strip()[:220]}")
  if kv:
    print("  KV cache tokens/replica:", ", ".join(sorted(kv)))
  for line in oom:
    print("  OOM", line)
  for line in err:
    print("  ERR", line)
  if os.path.exists(f"{d}/rewards.jsonl"):
    rs = [json.loads(line) for line in open(f"{d}/rewards.jsonl")]
    blocks = [rs[i:i + 10] for i in range(0, len(rs), 10)]
    print(f"  REWARD per 10 steps (n={len(rs)} steps):",
          [round(statistics.mean(r["acc"] for r in b), 3) for b in blocks])
  if os.path.exists(f"{d}/eval.jsonl"):
    for line in open(f"{d}/eval.jsonl"):
      r = json.loads(line)
      print("  EVAL", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
  main(sys.argv[1].rstrip("/"))
