#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

deepspeed --module src.cli.train_sft \
   --pretrain "${PRETRAIN_PATH:-Qwen/Qwen2.5-1.5B}" \
   --dataset "${SFT_DATASET:-openai/gsm8k}" \
   --dataset_split "${SFT_DATASET_SPLIT:-train}" \
   --input_key "${SFT_INPUT_KEY:-question}" \
   --output_key "${SFT_OUTPUT_KEY:-answer}" \
   --max_len "${SFT_MAX_LEN:-2048}" \
   --max_epochs "${SFT_MAX_EPOCHS:-3}" \
   --train_batch_size "${SFT_TRAIN_BATCH_SIZE:-128}" \
   --micro_train_batch_size "${SFT_MICRO_TRAIN_BATCH_SIZE:-8}" \
   --learning_rate "${SFT_LEARNING_RATE:-5e-6}" \
   --lr_scheduler "${SFT_LR_SCHEDULER:-cosine_with_min_lr}" \
   --lr_warmup_ratio "${SFT_LR_WARMUP_RATIO:-0.03}" \
   --save_path "${SFT_SAVE_PATH:-./ckpt/qwen2.5-1.5b-sft}" \
   --ckpt_path "${SFT_CKPT_PATH:-./ckpt/checkpoints_sft}" \
   --zero_stage "${ZERO_STAGE:-2}" \
   --param_dtype "${PARAM_DTYPE:-bf16}" \
   --gradient_checkpointing \
   --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
   --logging_steps "${SFT_LOGGING_STEPS:-10}" \
   --save_steps "${SFT_SAVE_STEPS:-50}" \
   --eval_steps "${SFT_EVAL_STEPS:--1}"
