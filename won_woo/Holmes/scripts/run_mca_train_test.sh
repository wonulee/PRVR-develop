#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN="${PYTHON_BIN:-/home/viglab2/miniconda3/envs/holmes/bin/python}"
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
exec "$PYTHON_BIN" -u -m Rewriter.train_experiment "$@"
