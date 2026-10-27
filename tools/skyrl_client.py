"""tinker-cookbook math_rl client with the shared eff/* efficiency metrics added to every step.

Usage (same CLI as tinker_cookbook.recipes.math_rl.train):
  python tools/skyrl_client.py base_url=... model_name=... env=polaris ...
"""
import asyncio
import os

import chz
import effmetrics
from tinker_cookbook.recipes.math_rl import train
from tinker_cookbook.utils import ml_log

_log_metrics = ml_log.MultiplexLogger.log_metrics


def _log_metrics_with_eff(self, metrics, step=None):
  if "time/total" in metrics and "env/all/total_ac_tokens" in metrics:
    metrics = {
        **metrics,
        **effmetrics.derive(
            step_time_s=metrics["time/total"],
            rollout_time_s=metrics.get("time/sampling"),
            train_time_s=metrics.get("time/train_step"),
            gen_tokens=metrics["env/all/total_ac_tokens"],
            prompt_tokens=metrics.get("env/all/total_ob_tokens"),
            samples=metrics.get("env/all/total_episodes"),
        ),
    }
  return _log_metrics(self, metrics, step)


ml_log.MultiplexLogger.log_metrics = _log_metrics_with_eff

if os.environ.get("SAVE_CKPT") == "0":
  # Profiling runs: skip the cookbook's unconditional final checkpoint (pass save_every=0 for the periodic ones).
  from tinker_cookbook import checkpoint_utils

  async def _no_final_save(self, loop_state):
    await self.finalize_async()
    return {}

  checkpoint_utils.CheckpointManager.save_final_async = _no_final_save

if __name__ == "__main__":
  asyncio.run(train.cli_main(chz.entrypoint(train.CLIConfig)))
