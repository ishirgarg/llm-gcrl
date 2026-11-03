"""Framework-agnostic TPU hardware monitor (sidecar, one per worker).

Polls the libtpu runtime metrics server (via tpu-info) and appends per-chip HBM usage and duty cycle
to a CSV. Started automatically by scripts/run_on_worker.sh; runs in the tools venv.

Usage: python hw_monitor.py <out.csv> [interval_s]
"""
import csv
import os
import sys
import time

from tpu_info import device, metrics


def main():
  out, interval = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
  chip_type, _ = device.get_local_chips()
  new_file = not os.path.exists(out)
  with open(out, "a", newline="") as f:
    w = csv.writer(f)
    if new_file:
      w.writerow(["time", "device_id", "hbm_used_bytes", "hbm_total_bytes", "duty_cycle_pct"])
    while True:
      try:
        usages = metrics.get_chip_usage(chip_type)
      except Exception:  # metrics server is only up while a TPU program is running
        usages = []
      now = time.time()
      for u in usages:
        w.writerow([f"{now:.1f}", u.device_id, u.memory_usage, u.total_memory, f"{u.duty_cycle_pct:.2f}"])
      f.flush()
      time.sleep(interval)


if __name__ == "__main__":
  main()
