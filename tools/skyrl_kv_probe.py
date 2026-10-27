"""Sampling-only probe against a running SkyRL Tinker server, to measure KV-cache memory and decode throughput.

For each completion length it samples `num_samples` sequences from one fixed prompt (twice: the first call
includes compilation) and appends {max_tokens, repeat, t0, t1, gen_tokens, prompt_tokens} to
$LOG_ROOT/$RUN_NAME/kv_probe.jsonl. tools/kv_report.py lines these windows up with the hw-*.csv HBM samples.

Usage: python skyrl_kv_probe.py --base-url http://127.0.0.1:8000 --model Qwen/Qwen3-4B [--lengths 512 1024 2048]
"""
import argparse
import json
import os
import time

import tinker
from tinker import types
from transformers import AutoTokenizer

PROMPT = (
    "Let $a$, $b$, $c$ be positive reals with $a+b+c=1$. Find the minimum of "
    "$\\frac{1}{a}+\\frac{4}{b}+\\frac{9}{c}$. Write your answer in \\boxed{} format."
)


def main():
  p = argparse.ArgumentParser()
  p.add_argument("--base-url", required=True)
  p.add_argument("--model", required=True)
  p.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048])
  p.add_argument("--num-samples", type=int, default=256)
  p.add_argument("--repeats", type=int, default=2)
  args = p.parse_args()

  out = os.path.join(os.environ["LOG_ROOT"], os.environ["RUN_NAME"], "kv_probe.jsonl")
  tok = AutoTokenizer.from_pretrained(args.model)
  text = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)
  ids = tok.encode(text, add_special_tokens=False)
  prompt = types.ModelInput.from_ints(list(ids))
  sampler = tinker.ServiceClient(base_url=args.base_url).create_sampling_client(base_model=args.model)

  for length in args.lengths:
    for rep in range(args.repeats):
      t0 = time.time()
      res = sampler.sample(prompt=prompt, num_samples=args.num_samples,
                           sampling_params=types.SamplingParams(max_tokens=length, temperature=1.0)).result()
      t1 = time.time()
      rec = {"max_tokens": length, "repeat": rep, "t0": t0, "t1": t1, "prompt_tokens": len(ids),
             "num_samples": args.num_samples, "gen_tokens": sum(len(s.tokens) for s in res.sequences)}
      print(json.dumps(rec), flush=True)
      with open(out, "a") as f:
        f.write(json.dumps(rec) + "\n")
      time.sleep(10)  # let the hw monitor record the idle level between probes


if __name__ == "__main__":
  main()
