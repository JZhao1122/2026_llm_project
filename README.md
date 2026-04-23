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

Evaluation:

```bash
bash eval_gsm8k.sh ./ckpt/qwen2.5-1.5b-sft
bash eval_mmlu.sh ./ckpt/qwen2.5-1.5b-sft
bash eval_all_sft_ckpts.sh
```

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
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `HF_ENDPOINT`, `HF_TOKEN`: Hugging Face mirror/auth settings when needed
