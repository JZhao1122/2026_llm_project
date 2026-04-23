#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

python -m src.cli.eval_sft_checkpoints \
  --ckpt_path "${SFT_CKPT_PATH:-./ckpt/checkpoints_sft}" \
  --save_path "${SFT_SAVE_PATH:-./ckpt/qwen2.5-1.5b-sft}" \
  --output_dir "${SFT_EVAL_OUTPUT_DIR:-./ckpt/eval_sft_checkpoints}" \
  --include_final \
  "$@"
