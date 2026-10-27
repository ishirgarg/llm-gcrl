"""Find the max context a SkyRL Tinker server handles without OOM, for rollout and for training.

  --mode sample: sample `num_samples` sequences at once with max_tokens = L (dense KV cache is allocated at the
                 rounded full length, so this is the worst case regardless of where samples stop)
  --mode train:  forward_backward on `num_samples` synthetic sequences of exactly L tokens with a LoRA model
                 (the server splits them into train_micro_batch_size micro-batches)
L steps through --lengths (default: SkyRL's dense-cache bucket sizes minus a small prompt) until a request fails. Each attempt is appended to
$LOG_ROOT/$RUN_NAME/maxctx.jsonl with timing; tools/kv_report.py-style HBM peaks come from hw-*.csv.
"""
import argparse
import json
import os
import random
import time

import tinker
from tinker import types


def log(rec):
  print(json.dumps(rec), flush=True)
  with open(os.path.join(os.environ["LOG_ROOT"], os.environ["RUN_NAME"], "maxctx.jsonl"), "a") as f:
    f.write(json.dumps(rec) + "\n")


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--base-url", required=True)
  p.add_argument("--model", required=True)
  p.add_argument("--mode", choices=["sample", "train"], required=True)
  p.add_argument("--num-samples", type=int, default=128)
  p.add_argument("--lengths", type=int, nargs="+",
                  default=[1920, 2944, 3968, 5888, 7936, 11904, 15872, 23808, 31744, 47616, 63488])
  p.add_argument("--lora-rank", type=int, default=32)
  args = p.parse_args()

  service = tinker.ServiceClient(base_url=args.base_url)
  prompt_ids = list(range(1000, 1069))  # 69-token prompt, like the KV probe
  if args.mode == "sample":
    client = service.create_sampling_client(base_model=args.model)
  else:
    client = service.create_lora_training_client(base_model=args.model, rank=args.lora_rank)

  for length in args.lengths:
    t0 = time.time()
    try:
      if args.mode == "sample":
        res = client.sample(prompt=types.ModelInput.from_ints(prompt_ids), num_samples=args.num_samples,
                            sampling_params=types.SamplingParams(max_tokens=length, temperature=1.0)).result()
        extra = {"gen_tokens": sum(len(s.tokens) for s in res.sequences)}
      else:
        rng = random.Random(length)
        data = []
        for _ in range(args.num_samples):
          toks = [rng.randrange(1000, 100000) for _ in range(length + 1)]
          data.append(types.Datum(
              model_input=types.ModelInput.from_ints(toks[:-1]),
              loss_fn_inputs={"target_tokens": toks[1:], "weights": [1.0] * length}))
        client.forward_backward(data, loss_fn="cross_entropy").result()
        extra = {"train_tokens": args.num_samples * length}
      log({"mode": args.mode, "length": length, "ok": True, "t0": t0, "t1": time.time(),
           "num_samples": args.num_samples, **extra})
    except Exception as e:  # OOM surfaces as a failed request from the server
      log({"mode": args.mode, "length": length, "ok": False, "t0": t0, "t1": time.time(),
           "num_samples": args.num_samples, "error": f"{type(e).__name__}: {str(e)[:300]}"})
      break
    time.sleep(10)


if __name__ == "__main__":
  main()
