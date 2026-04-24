#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
  if [[ "${GPU_COUNT}" -gt 0 ]]; then
    export CUDA_VISIBLE_DEVICES="$(seq -s, 0 $((GPU_COUNT - 1)))"
  fi
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
fi

VISIBLE_GPU_COUNT="$(python3 - <<'PY'
import os

cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
gpu_ids = [item.strip() for item in cuda_visible_devices.split(",") if item.strip()]
print(len(gpu_ids))
PY
)"
if [[ "${VISIBLE_GPU_COUNT}" -lt 1 ]]; then
  VISIBLE_GPU_COUNT=1
fi

GRPO_VLLM_NUM_ENGINES="${GRPO_VLLM_NUM_ENGINES:-1}"
GRPO_ACTOR_GPUS_PER_NODE="${GRPO_ACTOR_GPUS_PER_NODE:-${VISIBLE_GPU_COUNT}}"
GRPO_REF_GPUS_PER_NODE="${GRPO_REF_GPUS_PER_NODE:-${GRPO_ACTOR_GPUS_PER_NODE}}"
if [[ -n "${GRPO_VLLM_TP_SIZE:-}" ]]; then
  GRPO_VLLM_TP_SIZE="${GRPO_VLLM_TP_SIZE}"
else
  if (( VISIBLE_GPU_COUNT % GRPO_VLLM_NUM_ENGINES != 0 )); then
    echo "VISIBLE_GPU_COUNT=${VISIBLE_GPU_COUNT} must be divisible by GRPO_VLLM_NUM_ENGINES=${GRPO_VLLM_NUM_ENGINES}" >&2
    exit 1
  fi
  GRPO_VLLM_TP_SIZE="$((VISIBLE_GPU_COUNT / GRPO_VLLM_NUM_ENGINES))"
fi

EXTRA_ARGS=()
if [[ "${GRPO_COLOCATE_ALL_MODELS:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--colocate_all_models)
fi
if [[ "${GRPO_COLOCATE_ACTOR_REF:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--colocate_actor_ref)
fi
if [[ "${GRPO_SAVE_HF_CKPT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--save_hf_ckpt)
fi
if [[ "${GRPO_LOAD_CHECKPOINT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--load_checkpoint)
fi
if [[ "${GRPO_DISABLE_DS_CKPT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--disable_ds_ckpt)
fi
if [[ "${GRPO_USE_DS_UNIVERSAL_CKPT:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--use_ds_universal_ckpt)
fi
if [[ "${GRPO_VLLM_ENABLE_SLEEP:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--vllm_enable_sleep)
fi
if [[ "${GRPO_DEEPSPEED_ENABLE_SLEEP:-1}" == "1" ]]; then
  EXTRA_ARGS+=(--deepspeed_enable_sleep)
fi
if [[ "${GRPO_ENABLE_PREFIX_CACHING:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--enable_prefix_caching)
fi
if [[ "${GRPO_REF_REWARD_OFFLOAD:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--ref_reward_offload)
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
   --max_samples "${GRPO_MAX_SAMPLES:-20000}" \
   --actor_num_nodes "${GRPO_ACTOR_NUM_NODES:-1}" \
   --actor_num_gpus_per_node "${GRPO_ACTOR_GPUS_PER_NODE}" \
   --ref_num_nodes "${GRPO_REF_NUM_NODES:-1}" \
   --ref_num_gpus_per_node "${GRPO_REF_GPUS_PER_NODE}" \
   --vllm_num_engines "${GRPO_VLLM_NUM_ENGINES}" \
   --vllm_tensor_parallel_size "${GRPO_VLLM_TP_SIZE}" \
   --vllm_sync_backend "${GRPO_VLLM_SYNC_BACKEND:-nccl}" \
   --vllm_gpu_memory_utilization "${GRPO_VLLM_GPU_MEM_UTIL:-0.5}" \
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
   --advantage_estimator "${GRPO_ADVANTAGE_ESTIMATOR:-group_norm}" \
   --dist_backend "${GRPO_DIST_BACKEND:-nccl}" \
   --zero_stage "${ZERO_STAGE:-2}" \
   --param_dtype "${PARAM_DTYPE:-bf16}" \
   --gradient_checkpointing \
   --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
   --save_path "${GRPO_SAVE_PATH:-./ckpt/qwen2.5-1.5b-grpo}" \
   --ckpt_path "${GRPO_CKPT_PATH:-./ckpt/checkpoints_grpo}" \
   --save_steps "${GRPO_SAVE_STEPS:-20}" \
   --max_ckpt_num "${GRPO_MAX_CKPT_NUM:-3}" \
   --max_ckpt_mem "${GRPO_MAX_CKPT_MEM:-100000000}" \
   --logging_steps "${GRPO_LOGGING_STEPS:-1}" \
   "${EXTRA_ARGS[@]}"
