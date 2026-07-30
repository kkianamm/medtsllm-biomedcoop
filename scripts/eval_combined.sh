#!/usr/bin/env bash
set -euo pipefail
RUN_ID="${1:?Usage: eval_combined.sh RUN_ID [SPLIT] [CHECKPOINT]}"
SPLIT="${2:-test}"
CHECKPOINT="${3:-best}"
python3 test.py "$RUN_ID" "$SPLIT" "$CHECKPOINT"
