#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash eval_gsm8k.sh <YOUR_MODEL_PATH> [extra eval args...]"
  exit 1
fi

MODEL_PATH="$1"
shift

python -m src.cli.eval_model \
  --model_path "$MODEL_PATH" \
  --tasks gsm8k \
  --gsm8k_tensor_parallel_size "${GSM8K_TP_SIZE:-1}" \
  --gsm8k_batch_size "${GSM8K_BATCH_SIZE:-32}" \
  --gsm8k_gpu_memory_utilization "${GSM8K_GPU_MEMORY_UTILIZATION:-0.9}" \
  "$@"
