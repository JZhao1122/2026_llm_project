#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

python3 -m src.cli.train_grpo \
   --pretrain "${PRETRAIN_PATH:-Qwen/Qwen2.5-1.5B}" \
   --reward_fn "${REWARD_FN_PATH:-./reward_func_gsm8k.py}" \
   --prompt_data "${GRPO_PROMPT_DATA:-openai/gsm8k#main}" \
   --prompt_split "${GRPO_PROMPT_SPLIT:-train}" \
   --eval_dataset "${GRPO_EVAL_DATASET:-openai/gsm8k#main}" \
   --eval_split "${GRPO_EVAL_SPLIT:-test}" \
   --eval_steps "${GRPO_EVAL_STEPS:-5}" \
   --eval_n_samples_per_prompt "${GRPO_EVAL_SAMPLES_PER_PROMPT:-1}" \
   --input_key "${GRPO_INPUT_KEY:-question}" \
   --label_key "${GRPO_LABEL_KEY:-answer}" \
   --colocate_all_models \
   --max_samples "${GRPO_MAX_SAMPLES:-20000}" \
   --actor_num_nodes "${GRPO_ACTOR_NUM_NODES:-1}" \
   --actor_num_gpus_per_node "${GRPO_ACTOR_GPUS_PER_NODE:-2}" \
   --ref_num_nodes "${GRPO_REF_NUM_NODES:-1}" \
   --ref_num_gpus_per_node "${GRPO_REF_GPUS_PER_NODE:-2}" \
   --vllm_num_engines "${GRPO_VLLM_NUM_ENGINES:-1}" \
   --vllm_tensor_parallel_size "${GRPO_VLLM_TP_SIZE:-2}" \
   --vllm_gpu_memory_utilization "${GRPO_VLLM_GPU_MEM_UTIL:-0.5}" \
   --vllm_enable_sleep \
   --deepspeed_enable_sleep \
   --enforce_eager \
   --n_samples_per_prompt "${GRPO_SAMPLES_PER_PROMPT:-8}" \
   --rollout_batch_size "${GRPO_ROLLOUT_BATCH_SIZE:-64}" \
   --micro_rollout_batch_size "${GRPO_MICRO_ROLLOUT_BATCH_SIZE:-8}" \
   --train_batch_size "${GRPO_TRAIN_BATCH_SIZE:-128}" \
   --micro_train_batch_size "${GRPO_MICRO_TRAIN_BATCH_SIZE:-2}" \
   --prompt_max_len "${GRPO_PROMPT_MAX_LEN:-512}" \
   --generate_max_len "${GRPO_GENERATE_MAX_LEN:-512}" \
   --num_episodes "${GRPO_NUM_EPISODES:-1}" \
   --max_epochs "${GRPO_MAX_EPOCHS:-1}" \
   --actor_learning_rate "${GRPO_ACTOR_LR:-5e-7}" \
   --init_kl_coef "${GRPO_INIT_KL_COEF:-1e-3}" \
   --use_kl_loss \
   --kl_estimator "${GRPO_KL_ESTIMATOR:-k3}" \
   --eps_clip "${GRPO_EPS_CLIP:-0.2}" \
   --gamma "${GRPO_GAMMA:-1.0}" \
   --temperature "${GRPO_TEMPERATURE:-0.7}" \
   --zero_stage "${ZERO_STAGE:-2}" \
   --param_dtype "${PARAM_DTYPE:-bf16}" \
   --gradient_checkpointing \
   --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
   --save_path "${GRPO_SAVE_PATH:-./ckpt/qwen2.5-1.5b-grpo}" \
   --ckpt_path "${GRPO_CKPT_PATH:-./ckpt/checkpoints_grpo}" \
   --save_steps "${GRPO_SAVE_STEPS:-20}" \
   --logging_steps "${GRPO_LOGGING_STEPS:-1}"
