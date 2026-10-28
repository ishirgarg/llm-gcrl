"""MATH for RL validation runs (Tunix data_module): train on the MATH train split, levels 3-5 (the SimpleRL-Zoo /
Dr. GRPO zero-RL setup), evaluate on MATH-500 (held-out subset of the MATH test split), AMC23 and AIME24
(tools/rl_eval.py).

Prompt = SimpleRL-Zoo "qwen-boxed": the problem + "Please reason step by step, and put your final answer within
\\boxed{}." as a Qwen chat user turn. Base models get the chat format written out (no thinking tags); hybrid
thinking models get their own template with thinking disabled. Reward: tools/math_reward.py (last \\boxed{}).

Each element: {"prompts": templated prompt string, "question": problem, "answer": reference answer}.
"""
import os

import datasets
import grain
import math_reward
from transformers import AutoTokenizer

INSTRUCTION = "\nPlease reason step by step, and put your final answer within \\boxed{}."
_SYSTEM = "You are a helpful assistant."


def _model_id():
  return os.environ.get("MODEL", "Qwen/Qwen3-4B")


def make_prompt(problem, model_id=None):
  model_id = model_id or _model_id()
  user = problem + INSTRUCTION
  if "base" in model_id.lower():
    return (f"<|im_start|>system\n{_SYSTEM}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n")
  tok = AutoTokenizer.from_pretrained(model_id)
  return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                 enable_thinking=False)


def _answer_from_solution(solution):
  try:
    return math_reward.grading.extract_boxed(solution)
  except ValueError:
    return None


def create_dataset(model_id=None, seed=0):
  model_id = model_id or _model_id()
  seed = int(os.environ.get("DATA_SEED", seed))
  levels = {f"Level {l}" for l in os.environ.get("MATH_LEVELS", "3,4,5").split(",")}
  ds = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", "default", split="train")
  ds = ds.filter(lambda x: x["level"] in levels)
  ds = ds.map(lambda x: {"answer": _answer_from_solution(x["solution"])})
  ds = ds.filter(lambda x: bool(x["answer"])).shuffle(seed=seed)
  template = make_prompt("{PROBLEM}", model_id)

  def to_prompt(x):
    return {"prompts": template.replace("{PROBLEM}", x["problem"]), "question": x["problem"],
            "answer": str(x["answer"])}

  return grain.MapDataset.source(ds).map(to_prompt)


_EVAL_SETS = {  # name -> (HF dataset, split, problem column, answer column)
    "math500": ("HuggingFaceH4/MATH-500", "test", "problem", "answer"),
    "amc23": ("math-ai/amc23", "test", "question", "answer"),
    "aime24": ("HuggingFaceH4/aime_2024", "train", "problem", "answer"),
}


def eval_set(name, model_id=None):
  """[(prompt, answer)] for one evaluation set."""
  path, split, q, a = _EVAL_SETS[name]
  ds = datasets.load_dataset(path, split=split)
  template = make_prompt("{PROBLEM}", model_id)
  return [(template.replace("{PROBLEM}", x[q]), str(x[a])) for x in ds]
