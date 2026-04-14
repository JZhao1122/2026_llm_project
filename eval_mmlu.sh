#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash eval_mmlu.sh <YOUR_MODEL_PATH> [extra eval args...]"
  exit 1
fi

MODEL_PATH="$1"
shift

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON="$PYTHON_BIN"
elif [[ -x "/Users/user/mambaforge/bin/python" ]]; then
  PYTHON="/Users/user/mambaforge/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON="$(command -v python)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON="$(command -v python3)"
else
  echo "Error: could not find a Python interpreter." >&2
  exit 1
fi

"$PYTHON" -m openrlhf.cli.eval_model \
  --model_path "$MODEL_PATH" \
  --tasks mmlu \
  --mmlu_batch_size "${MMLU_BATCH_SIZE:-16}" \
  --mmlu_device_map "${MMLU_DEVICE_MAP:-auto}" \
  "$@"
