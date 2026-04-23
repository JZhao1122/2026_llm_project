import argparse
import math
import os

from transformers.trainer import get_scheduler

from ..datasets import SFTDataset
from ..datasets.utils import blending_datasets
from ..models import SFTModel
from ..trainer.sft_trainer import SFTTrainer
from ..utils import get_strategy, get_tokenizer


def train(args):
    strategy = get_strategy(args)
    strategy.setup_distributed()

    model = SFTModel(
        args.pretrain,
        attn_implementation=args.attn_implementation,
        param_dtype=args.param_dtype,
        ds_config=strategy.get_ds_train_config(is_actor=True),
    )
    tokenizer = get_tokenizer(args.pretrain, model.model, "right")
    strategy.print(model)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optim = strategy.create_optimizer(model, lr=args.learning_rate, betas=args.adam_betas, weight_decay=args.l2)

    train_data = blending_datasets(
        args.dataset,
        args.dataset_probs,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.dataset_split,
        is_rank_0=strategy.is_rank_0(),
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))

    eval_data = None
    if getattr(args, "eval_dataset", None):
        eval_data = blending_datasets(
            args.eval_dataset,
            None,
            dataset_split=args.eval_split,
            is_rank_0=strategy.is_rank_0(),
        )
    elif args.eval_ratio > 0:
        if not 0 < args.eval_ratio < 1:
            raise ValueError(f"eval_ratio must be in (0, 1), got {args.eval_ratio}")
        split_dataset = train_data.train_test_split(test_size=args.eval_ratio, seed=args.seed, shuffle=True)
        train_data = split_dataset["train"]
        eval_data = split_dataset["test"]
        strategy.print(
            f"Split {args.dataset} into train/eval with eval_ratio={args.eval_ratio}: "
            f"{len(train_data)} train / {len(eval_data)} eval samples"
        )

    train_dataset = SFTDataset(
        train_data,
        tokenizer,
        args.max_len,
        input_key=args.input_key,
        output_key=args.output_key,
        prompt_template=args.prompt_template,
        apply_chat_template=args.apply_chat_template,
        tokenizer_chat_template=args.tokenizer_chat_template,
        multiturn=args.multiturn,
    )
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.micro_train_batch_size,
        True,
        True,
        train_dataset.collate_fn,
    )

    eval_dataloader = None
    if eval_data is not None:
        eval_dataset = SFTDataset(
            eval_data,
            tokenizer,
            args.max_len,
            input_key=args.input_key,
            output_key=args.output_key,
            prompt_template=args.prompt_template,
            apply_chat_template=args.apply_chat_template,
            tokenizer_chat_template=args.tokenizer_chat_template,
            multiturn=args.multiturn,
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.micro_train_batch_size,
            True,
            False,
            eval_dataset.collate_fn,
        )

    num_update_steps_per_epoch = len(train_dataset) // args.train_batch_size
    max_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.max_epochs * num_update_steps_per_epoch)
    scheduler = get_scheduler(
        args.lr_scheduler,
        optim,
        num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
        num_training_steps=max_steps,
        scheduler_specific_kwargs={"min_lr": args.learning_rate * 0.1},
    )

    model, optim, scheduler = strategy.prepare((model, optim, scheduler))

    consumed_samples = 0
    if args.load_checkpoint and os.path.exists(args.ckpt_path):
        _, states = strategy.load_ckpt(model.model, args.ckpt_path)
        consumed_samples = states["consumed_samples"]
        strategy.print(f"Loaded the checkpoint: {args.ckpt_path}, consumed_samples: {consumed_samples}")

    os.makedirs(args.save_path, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        strategy=strategy,
        optim=optim,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
        scheduler=scheduler,
        batch_size=args.train_batch_size,
        max_epochs=args.max_epochs,
        max_steps=max_steps,
        tokenizer=tokenizer,
        save_hf_ckpt=args.save_hf_ckpt,
        disable_ds_ckpt=args.disable_ds_ckpt,
    )

    trainer.fit(args, consumed_samples, num_update_steps_per_epoch)
    strategy.save_model(model, tokenizer, args.save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_sft")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=100000000)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    parser.add_argument("--micro_train_batch_size", type=int, default=8, help="batch size per GPU")
    parser.add_argument("--train_batch_size", type=int, default=128, help="global training batch size")
    parser.add_argument("--max_norm", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--zero_stage", type=int, default=2, help="DeepSpeed ZeRO stage")
    parser.add_argument("--param_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    parser.add_argument("--max_epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=-1, help="Override total optimizer steps; -1 uses epochs.")
    parser.add_argument("--pretrain", type=str, default=None)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--l2", type=float, default=0, help="weight decay")
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95), help="Betas for Adam")

    parser.add_argument("--dataset", type=str, default=None, help="Path to the training dataset")
    parser.add_argument("--dataset_probs", type=str, default=None, help="Sampling probabilities for training datasets")
    parser.add_argument("--eval_dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument("--dataset_split", type=str, default="train")
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument(
        "--eval_ratio",
        type=float,
        default=0.0,
        help="If eval_dataset is unset, split this fraction from the loaded training data as validation.",
    )
    parser.add_argument("--max_samples", type=int, default=1000000, help="Maximum number of samples to use")
    parser.add_argument("--multiturn", action="store_true", default=False, help="Use multiturn conversation data")

    parser.add_argument("--input_key", type=str, default="input")
    parser.add_argument("--output_key", type=str, default="output")
    parser.add_argument("--prompt_template", type=str, default="User: {}\nAssistant: ")
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)
    parser.add_argument("--max_len", type=int, default=2048, help="Max tokens per sample")

    args = parser.parse_args()

    if args.prompt_template and "{}" not in args.prompt_template:
        print("[Warning] {} not in args.prompt_template, set to None")
        args.prompt_template = None

    if args.prompt_template and "\\n" in args.prompt_template:
        print(
            "[Warning] prompt_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    train(args)
