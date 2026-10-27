#!/bin/bash
# Launch with: scripts/tpu.sh launch check scripts/check_tpu.sh [conda-env]
activate_env "${1:-llm-gcrl}"
python "$REPO/tools/check_tpu.py"
