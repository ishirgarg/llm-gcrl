"""Countdown as a goal-conditioned task: reach a target number from 4 given numbers with + - * / (each used once).

A goal is just (numbers, target), so any goal can be proposed and scored:

  goal = Goal((3, 7, 25, 50), 123)
  prompt(goal)                         chat-templated prompt for the policy
  achieved(goal, completion)           integer value of the answer expression, or None if it is not a valid
                                       expression over exactly these numbers
  reward(goal, completion)             1.0 if achieved == target, else 0.0
  Goal(goal.numbers, achieved(...))    hindsight relabeling: a wrong but valid answer solves the goal it reached

The prompt asks for the closest expression when the target is out of reach, so (almost) every rollout reaches some
goal that hindsight relabeling can use; the reward is still exact-match only.

Tunix data module: create_dataset() yields training goals; eval_set(name) gives held-out goals for tools/rl_eval.py.
Goals are generated from seeds (no download): random numbers, combined in a random order with random operations
whose intermediate results stay positive integers, so every target is reachable. COUNTDOWN_TARGETS="lo-hi" sets
the target range (default 10-100, as in TinyZero's Countdown-Tasks).
"""
import ast
import dataclasses
import fractions
import functools
import os
import random
import re

N_NUMBERS = 4
NUMBER_RANGE = (1, 99)


def target_range():
  lo, hi = os.environ.get("COUNTDOWN_TARGETS", "10-100").split("-")
  return int(lo), int(hi)


@dataclasses.dataclass(frozen=True)
class Goal:
  numbers: tuple[int, ...]
  target: int

  def encode(self):  # dataset fields are strings, which survive Tunix's batching unchanged
    return {"numbers": " ".join(map(str, self.numbers)), "target": str(self.target)}

  @classmethod
  def decode(cls, numbers, target):
    return cls(tuple(int(x) for x in str(numbers).split()), int(target))


# ----------------------------------------------------------------------------------------------- prompting

USER_TEMPLATE = (
    "Using the numbers {numbers}, create an equation that equals {target}. You can use +, -, *, / and "
    "parentheses, and each number must be used exactly once. Think step by step, then write only the final "
    "expression (without \"= {target}\") inside <answer> </answer> tags, for example <answer>(1 + 2) * 3 - 4</answer>. "
    "If you cannot reach {target} exactly, answer with the closest expression you found."
)


@functools.lru_cache(maxsize=None)
def _chat_template(model_id):
  """The model's chat template around a {USER} placeholder (thinking disabled for hybrid models)."""
  from transformers import AutoTokenizer  # pylint: disable=g-import-not-at-top
  tok = AutoTokenizer.from_pretrained(model_id)
  return tok.apply_chat_template([{"role": "user", "content": "{USER}"}], tokenize=False,
                                 add_generation_prompt=True, enable_thinking=False)


def prompt(goal, model_id=None):
  user = USER_TEMPLATE.format(numbers=list(goal.numbers), target=goal.target)
  return _chat_template(model_id or os.environ.get("MODEL", "Qwen/Qwen3-4B-Instruct-2507")).replace("{USER}", user)


# ----------------------------------------------------------------------------------------------- scoring

_ANSWER = re.compile(r"answer>(.*?)</answer>", re.DOTALL)  # also "<|im_start|>answer>", a frequent slip
_SYMBOLS = {"×": "*", "\\times": "*", "·": "*", "÷": "/", "\\div": "/", "−": "-", "–": "-", "\\left": "", "\\right": ""}
_OPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b}


def answer_expression(completion):
  """The expression in the last <answer> tag, without any trailing '= ...' (None if there is no tag)."""
  found = _ANSWER.findall(completion)
  if not found:
    return None
  expr = found[-1]
  for s, r in _SYMBOLS.items():
    expr = expr.replace(s, r)
  return expr.split("=")[0].strip()


def _evaluate(node, used):
  """Exact value of an expression tree of + - * / over integer literals; collects the literals in `used`."""
  if isinstance(node, ast.Constant) and type(node.value) is int:  # pylint: disable=unidiomatic-typecheck
    used.append(node.value)
    return fractions.Fraction(node.value)
  if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
    a, b = _evaluate(node.left, used), _evaluate(node.right, used)
    if isinstance(node.op, ast.Div) and b == 0:
      raise ZeroDivisionError
    return _OPS[type(node.op)](a, b)
  raise ValueError("not a + - * / expression over integers")


def achieved(goal, completion):
  """The integer the completion's answer evaluates to, if it is a valid expression over exactly goal.numbers."""
  return value(goal, answer_expression(completion))


def value(goal, expr):
  """The integer value of expression text over exactly goal.numbers (None if invalid or not an integer)."""
  if not expr or len(expr) > 200:
    return None
  for s, r in _SYMBOLS.items():
    expr = expr.replace(s, r)
  used = []
  try:
    exact = _evaluate(ast.parse(expr.strip(), mode="eval").body, used)
  except (SyntaxError, ValueError, ZeroDivisionError, RecursionError):
    return None
  if sorted(used) != sorted(goal.numbers) or exact.denominator != 1:
    return None
  return int(exact)


def reward(goal, completion):
  return float(achieved(goal, completion) == goal.target)


# ----------------------------------------------------------------------------------------------- goals

def sample_goal(rng, targets=None):
  """Random reachable goal: combine N_NUMBERS random numbers pairwise with random valid operations."""
  lo, hi = targets or target_range()
  while True:
    numbers = [rng.randint(*NUMBER_RANGE) for _ in range(N_NUMBERS)]
    pool = list(numbers)
    while len(pool) > 1:
      a, b = sorted(rng.sample(pool, 2), reverse=True)
      pool.remove(a)
      pool.remove(b)
      results = [a + b, a * b] + ([a - b] if a > b else []) + ([a // b] if b > 1 and a % b == 0 else [])
      pool.append(rng.choice(results))
    if lo <= pool[0] <= hi:
      return Goal(tuple(numbers), pool[0])


def _goals(n, seed, targets=None):
  rng = random.Random(seed)  # string seeds hash deterministically, so every host builds the same goals
  return [sample_goal(rng, targets) for _ in range(n)]


def create_dataset(model_id=None, seed=0, size=20_000):
  import grain  # pylint: disable=g-import-not-at-top
  seed = int(os.environ.get("DATA_SEED", seed))
  goals = _goals(size, f"countdown-train-{seed}")
  rows = [{"prompts": prompt(g, model_id), **g.encode()} for g in goals]
  return grain.MapDataset.source(rows)


def eval_set(name, model_id=None):
  """[(prompt, answer)] for 'countdown' (training target range) or 'countdown_<lo>_<hi>' (another range).

  256 goals from a seed disjoint from training. The answer string is the encoded goal ("3 7 25 50|123").
  """
  parts = name.split("_")
  targets = (int(parts[1]), int(parts[2])) if len(parts) == 3 else None
  goals = _goals(256, "countdown-eval", targets)
  return [(prompt(g, model_id), f"{g.encode()['numbers']}|{g.target}") for g in goals]


def score(completion, answer):
  """rl_eval interface: 1.0 if correct, 0.0 if a valid but wrong expression or no answer."""
  numbers, target = answer.split("|")
  return reward(Goal.decode(numbers, target), completion)


def formatted(completion):
  """rl_eval interface: the completion has an <answer> tag."""
  return answer_expression(completion) is not None
