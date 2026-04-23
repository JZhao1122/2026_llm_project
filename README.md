# 2026_llm_project

Minimal local training tree for the IIIS 2026 LLM project.

The active code lives under `src/`. `student_v0/` is an archived baseline snapshot and is not used by the main scripts.

## What This Repo Contains

- `run_sft.sh`: launch supervised fine-tuning with `deepspeed --module src.cli.train_sft`
- `run_grpo.sh`: launch GRPO with `python -m src.cli.train_grpo`
- `eval_gsm8k.sh` / `eval_mmlu.sh`: standalone evaluation entrypoints
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

The default SFT dataset string uses the explicit config form `openai/gsm8k#main`.

GRPO:

```bash
PRETRAIN_PATH=./ckpt/qwen2.5-1.5b-sft HF_TOKEN=... bash run_grpo.sh
```

Evaluation:

```bash
bash eval_gsm8k.sh ./ckpt/qwen2.5-1.5b-sft
bash eval_mmlu.sh ./ckpt/qwen2.5-1.5b-sft
```

## Key Runtime Variables

- `PRETRAIN_PATH`: base model id or local checkpoint path
- `SFT_SAVE_PATH`, `SFT_CKPT_PATH`, `SFT_SAVE_STEPS`: SFT output and checkpoint cadence
- `GRPO_SAVE_PATH`, `GRPO_CKPT_PATH`, `GRPO_SAVE_STEPS`: GRPO output and checkpoint cadence
- `CUDA_VISIBLE_DEVICES`: GPU selection
- `HF_ENDPOINT`, `HF_TOKEN`: Hugging Face mirror/auth settings when needed
