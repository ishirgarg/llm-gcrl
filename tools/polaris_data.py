"""POLARIS-53K for Tunix (data_module), with the same prompt as the SkyRL / tinker-cookbook run:
user message = problem + " Write your answer in \\boxed{} format.", Qwen3 chat template, thinking disabled.

Each element: {"prompts": templated prompt string, "question": problem, "answer": reference answer}.
"""
import os
import datasets
import grain
from transformers import AutoTokenizer

QUESTION_SUFFIX = " Write your answer in \\boxed{} format."


def create_dataset(model_id: str = "Qwen/Qwen3-4B", seed: int = 0) -> grain.MapDataset:
  seed = int(os.environ.get("DATA_SEED", seed))  # launch with DATA_SEED=N to change the data order
  tok = AutoTokenizer.from_pretrained(model_id)
  ds = datasets.load_dataset("POLARIS-Project/Polaris-Dataset-53K", split="train").shuffle(seed=seed)
  ds = ds.filter(lambda x: bool(x["problem"]) and bool(x["answer"]))

  def to_prompt(x):
    messages = [{"role": "user", "content": x["problem"] + QUESTION_SUFFIX}]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return {"prompts": prompt, "question": x["problem"], "answer": str(x["answer"])}

  return grain.MapDataset.source(ds).map(to_prompt)
