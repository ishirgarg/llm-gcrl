#!/bin/bash
# Launch with: scripts/tpu.sh launch check-lc scripts/check_local_collective.sh
source "$NFS_ROOT/venvs/maxtext/bin/activate"
python "$REPO/tools/check_local_collective.py"
