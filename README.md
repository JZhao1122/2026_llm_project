# 2026_llm_project

Minimal local training tree for the IIIS 2026 LLM project.

The active code lives under `src/`. `student_v0/` is an archived baseline snapshot and is not used by the main scripts.

## What This Repo Contains

- `run_sft.sh`: launch supervised fine-tuning with `deepspeed --module src.cli.train_sft`
- `run_grpo.sh`: launch GRPO with `python -m src.cli.train_grpo`
- `eval_gsm8k.sh` / `eval_mmlu.sh`: standalone evaluation entrypoints
- `eval_all_sft_ckpts.sh`: evaluate every saved SFT HF checkpoint plus the final export
- `reward_func_gsm8k.py`: rule-based reward for GSM8K-style exact-match checking
- `src/`: local package that mirrors the required training, model, dataset, and utility code

## Environment Assumptions

- Python environment already has the required runtime dependencies, including `torch`, `transformers`, `deepspeed`, `ray`, `vllm`, and related libraries.
- Model paths should be provided through environment variables when using local checkpoints or fixed server-side model mirrors.
- No repo-internal script relies on a hardcoded repo path.

## Common Commands

SFT:

```bash
PRETRAIN_PATH=/root/workspace/_hf_models/Qwen/Qwen2.5-1.5B bash run_sft.sh
```

The default SFT dataset string uses the explicit config form `openai/gsm8k#main`. By default, the script:

- uses all visible GPUs unless `CUDA_VISIBLE_DEVICES` is set explicitly
- trains for `8` epochs
- splits `5%` of the loaded training set into validation via `--eval_ratio 0.05`
- runs validation every `50` steps
- saves intermediate HF checkpoints at each save step via `--save_hf_ckpt`

To stop at a fixed optimizer step and retain every checkpoint directory, override the defaults explicitly, for example:

```bash
SFT_MAX_STEPS=300 \
SFT_MAX_CKPT_NUM=1000 \
SFT_MAX_CKPT_MEM=100000000 \
bash run_sft.sh
```

GRPO:

```bash
PRETRAIN_PATH=./ckpt/qwen2.5-1.5b-sft HF_TOKEN=... bash run_grpo.sh
```

By default, `run_grpo.sh` uses all visible GPUs, maps them into the colocated actor/ref/vLLM setup, saves DeepSpeed checkpoints every `GRPO_SAVE_STEPS`, and writes the final HF export to `GRPO_SAVE_PATH`. To also save intermediate HF checkpoints and retain more DeepSpeed checkpoint directories, override the defaults explicitly, for example:

```bash
PRETRAIN_PATH=./ckpt/checkpoints_sft_step200_keepall/global_step100_hf \
GRPO_SAVE_HF_CKPT=1 \
GRPO_MAX_CKPT_NUM=1000 \
GRPO_SAVE_STEPS=10 \
bash run_grpo.sh
```

If the pod has a very small `/dev/shm`, NCCL can fail during Ray/DeepSpeed startup. In that case, export `NCCL_SHM_DISABLE=1` before launching GRPO so the setting propagates into Ray workers as well.

If you need to avoid multi-GPU NCCL setup on the actor/reference side, disable `GRPO_COLOCATE_ALL_MODELS` and assign GPUs per role explicitly, for example `1` GPU for actor, `1` for reference, and `2` single-GPU vLLM engines.

When actor/reference each run on a single GPU and the pod still has NCCL shared-memory issues, set `GRPO_DIST_BACKEND=gloo` so DeepSpeed actor initialization avoids NCCL.

For a more stable 4-GPU layout, keep actor/reference on two GPUs and dedicate the other two GPUs to vLLM. Lowering `GRPO_VLLM_GPU_MEM_UTIL` to `0.3` leaves more headroom for KV-cache wakeups at the cost of rollout throughput:

```bash
PRETRAIN_PATH=/path/to/base-or-sft \
GRPO_COLOCATE_ALL_MODELS=0 \
GRPO_COLOCATE_ACTOR_REF=1 \
GRPO_ACTOR_GPUS_PER_NODE=2 \
GRPO_REF_GPUS_PER_NODE=2 \
GRPO_VLLM_NUM_ENGINES=2 \
GRPO_VLLM_TP_SIZE=1 \
GRPO_VLLM_GPU_MEM_UTIL=0.3 \
GRPO_GENERATE_MAX_LEN=256 \
GRPO_ROLLOUT_BATCH_SIZE=32 \
GRPO_MICRO_ROLLOUT_BATCH_SIZE=4 \
bash run_grpo.sh
```

Evaluation:

```bash
bash eval_gsm8k.sh ./ckpt/qwen2.5-1.5b-sft
bash eval_mmlu.sh ./ckpt/qwen2.5-1.5b-sft
bash eval_all_sft_ckpts.sh
```

By default, `eval_gsm8k.sh` now uses the raw `question` text as the prompt so it matches the current GRPO training setup (`input_key=question` with no prompt template). It also uses `temperature=0.6` and runs `3` sampled repeats, then reports the mean accuracy in `gsm8k.accuracy`. The per-repeat accuracies are also saved in the JSON output as `repeat_accuracies`, and the standard deviation is saved as `std_accuracy`.

To switch back to the original benchmark-style GSM8K instruction prompt, set `GSM8K_PROMPT_MODE=benchmark`. If you need a custom prompt wrapper, pass `--gsm8k_prompt_template '...{}...'` through `eval_gsm8k.sh`.

## Key Runtime Variables

- `PRETRAIN_PATH`: base model id or local checkpoint path
- `SFT_SAVE_PATH`, `SFT_CKPT_PATH`, `SFT_SAVE_STEPS`: SFT output and checkpoint cadence
- `SFT_MAX_EPOCHS`, `SFT_MAX_STEPS`, `SFT_EVAL_RATIO`, `SFT_EVAL_STEPS`: SFT training duration and validation split/cadence
- `SFT_MAX_CKPT_NUM`, `SFT_MAX_CKPT_MEM`: checkpoint retention limits passed to DeepSpeed
- `SFT_EVAL_OUTPUT_DIR`: output directory for `eval_all_sft_ckpts.sh`
- `GRPO_SAVE_PATH`, `GRPO_CKPT_PATH`, `GRPO_SAVE_STEPS`: GRPO output and checkpoint cadence
- `GRPO_SAVE_HF_CKPT`: set to `1` to additionally save `global_step*_hf` GRPO checkpoints
- `GRPO_MAX_CKPT_NUM`, `GRPO_MAX_CKPT_MEM`: DeepSpeed retention limits for GRPO checkpoints
- `GRPO_ACTOR_GPUS_PER_NODE`, `GRPO_REF_GPUS_PER_NODE`, `GRPO_VLLM_NUM_ENGINES`, `GRPO_VLLM_TP_SIZE`: GRPO parallelism layout across visible GPUs
- `GRPO_VLLM_GPU_MEM_UTIL`: vLLM memory reservation target; reduce it when wake-up OOMs appear in colocated or tight layouts
- `GRPO_COLOCATE_ALL_MODELS`, `GRPO_COLOCATE_ACTOR_REF`: toggle colocated GRPO layouts when the pod topology requires it
- `GRPO_VLLM_ENABLE_SLEEP`, `GRPO_DEEPSPEED_ENABLE_SLEEP`: toggle the vLLM/DeepSpeed sleep path used by colocated layouts
- `GRPO_VLLM_SYNC_BACKEND`: backend used for actor-to-vLLM weight sync when CUDA IPC is not active
- `GRPO_DIST_BACKEND`: distributed backend for DeepSpeed actor/reference init, useful to switch to `gloo` on single-GPU roles
- `GRPO_REF_REWARD_OFFLOAD`: enable DeepSpeed offload for reference/reward model evaluation states when memory is tight
- `GRPO_ENABLE_PREFIX_CACHING`: opt into vLLM prefix caching if repeated prompt prefixes justify the extra cache usage
- `NCCL_SHM_DISABLE`: set to `1` on pods with tiny `/dev/shm` to avoid NCCL shared-memory startup failures
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `HF_ENDPOINT`, `HF_TOKEN`: Hugging Face mirror/auth settings when needed
