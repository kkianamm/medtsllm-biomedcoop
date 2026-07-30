#!/usr/bin/env bash
set -euo pipefail
CONFIG="${1:-configs/combined_ptbxl.toml}"
RUN_ID="${2:-combined_ptbxl}"
python3 train.py "$CONFIG" "$RUN_ID"
