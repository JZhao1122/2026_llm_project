#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash eval_gsm8k.sh <YOUR_MODEL_PATH> [extra eval args...]"
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
  --tasks gsm8k \
  --gsm8k_tensor_parallel_size "${GSM8K_TP_SIZE:-1}" \
  --gsm8k_batch_size "${GSM8K_BATCH_SIZE:-32}" \
  --gsm8k_gpu_memory_utilization "${GSM8K_GPU_MEMORY_UTILIZATION:-0.9}" \
  "$@"
