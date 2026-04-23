#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash eval_mmlu.sh <YOUR_MODEL_PATH> [extra eval args...]"
  exit 1
fi

MODEL_PATH="$1"
shift

python -m src.cli.eval_model \
  --model_path "$MODEL_PATH" \
  --tasks mmlu \
  --mmlu_batch_size "${MMLU_BATCH_SIZE:-16}" \
  --mmlu_device_map "${MMLU_DEVICE_MAP:-none}" \
  --mmlu_device "${MMLU_DEVICE:-cuda}" \
  "$@"
