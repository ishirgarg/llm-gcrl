#!/bin/bash
# Launch with: scripts/tpu.sh launch check-local scripts/check_tpu_local.sh   (each host runs independently)
source "$REPO/scripts/single_host_env.sh"
activate_env llm-gcrl
python "$REPO/tools/check_tpu_local.py"
