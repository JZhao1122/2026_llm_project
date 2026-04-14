import argparse
import os

import ray
from ray.util.placement_group import placement_group

from ..trainer.grpo_trainer import RayGRPOTrainer, prepare_datasets
from ..trainer.ray import create_vllm_engines
from ..trainer.ray.grpo_actor import (
    GRPOPolicyModelActor,
    GRPOReferenceModelActor,
    GRPORewardModelActor,
)
from ..trainer.ray.launcher import RayActorGroup
from ..utils import get_strategy, get_tokenizer


def train(args):
    if not ray.is_initialized():
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    strategy = get_strategy(args)
    strategy.print(args)
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.ckpt_path, exist_ok=True)

    tokenizer = get_tokenizer(args.pretrain, None, "left")
    train_dataloader, eval_dataloader, max_steps = prepare_datasets(args, strategy, tokenizer)

    actor_ref_pg = None
    if args.colocate_actor_ref or args.colocate_all_models:
        if args.init_kl_coef > 0 or args.use_kl_loss:
            assert args.actor_num_nodes == args.ref_num_nodes
            assert args.actor_num_gpus_per_node == args.ref_num_gpus_per_node
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor_num_nodes * args.actor_num_gpus_per_node)]
        actor_ref_pg = placement_group(bundles, strategy="PACK")
        ray.get(actor_ref_pg.ready())

    if args.colocate_all_models:
        expected_gpus = args.vllm_num_engines * args.vllm_tensor_parallel_size
        actual_gpus = args.actor_num_nodes * args.actor_num_gpus_per_node
        assert (
            actual_gpus == expected_gpus
        ), f"actor GPUs ({actual_gpus}) must match vLLM GPUs ({expected_gpus}) when --colocate_all_models is enabled"

    vllm_engines = create_vllm_engines(
        args.vllm_num_engines,
        args.vllm_tensor_parallel_size,
        args.pretrain,
        args.seed,
        args.full_determinism,
        args.enable_prefix_caching,
        args.enforce_eager,
        args.prompt_max_len + args.generate_max_len,
        actor_ref_pg if args.colocate_all_models else None,
        args.vllm_gpu_memory_utilization,
        args.vllm_enable_sleep,
        args.deepspeed_enable_sleep,
        remote_rm_url=args.remote_rm_url,
    )

    actor_model = RayActorGroup(
        args.actor_num_nodes,
        args.actor_num_gpus_per_node,
        GRPOPolicyModelActor,
        pg=actor_ref_pg if (args.colocate_actor_ref or args.colocate_all_models) else None,
        num_gpus_per_actor=0.2 if actor_ref_pg else 1,
    )

    reference_model = None
    if args.init_kl_coef > 0 or args.use_kl_loss:
        reference_model = RayActorGroup(
            args.ref_num_nodes,
            args.ref_num_gpus_per_node,
            GRPOReferenceModelActor,
            pg=actor_ref_pg if (args.colocate_actor_ref or args.colocate_all_models) else None,
            num_gpus_per_actor=0.2 if actor_ref_pg else 1,
        )

    reward_model = None
    if args.remote_rm_url is None:
        reward_model = RayActorGroup(
            args.reward_num_nodes,
            args.reward_num_gpus_per_node,
            GRPORewardModelActor,
        )

    refs = []
    refs.extend(actor_model.async_init_model_from_pretrained(strategy, args.pretrain, max_steps, vllm_engines))
    if reference_model is not None:
        refs.extend(reference_model.async_init_model_from_pretrained(strategy, args.pretrain))
    if reward_model is not None:
        refs.extend(reward_model.async_init_model_from_pretrained(strategy, args.reward_pretrain))
    ray.get(refs)

    trainer = RayGRPOTrainer(
        strategy=strategy,
        actor_model_group=actor_model,
        reference_model_group=reference_model,
        reward_model_group=reward_model,
        vllm_engines=vllm_engines,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        eval_dataloader=eval_dataloader,
    )
    trainer.fit()
    ray.get(actor_model.async_save_model())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--actor_num_nodes", type=int, default=1)
    parser.add_argument("--actor_num_gpus_per_node", type=int, default=8)
    parser.add_argument("--ref_num_nodes", type=int, default=1)
    parser.add_argument("--ref_num_gpus_per_node", type=int, default=8)
    parser.add_argument("--reward_num_nodes", type=int, default=1)
    parser.add_argument("--reward_num_gpus_per_node", type=int, default=8)
    parser.add_argument("--colocate_actor_ref", action="store_true", default=False)
    parser.add_argument("--colocate_all_models", action="store_true", default=False)

    parser.add_argument("--vllm_num_engines", type=int, required=True)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument("--vllm_sync_backend", type=str, default="nccl")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--vllm_enable_sleep", action="store_true", default=False)
    parser.add_argument(
        "--deepspeed_enable_sleep",
        action="store_true",
        default=False,
        help="Enable DeepSpeed state offload/reload for GRPO colocated sleep mode.",
    )
    parser.add_argument("--enable_prefix_caching", action="store_true", default=False)
    parser.add_argument("--enforce_eager", action="store_true", default=False)

    parser.add_argument("--save_path", type=str, default="./ckpt")
    parser.add_argument("--save_steps", type=int, default=-1)
    parser.add_argument("--save_hf_ckpt", action="store_true", default=False)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=-1)
    parser.add_argument("--ckpt_path", type=str, default="./ckpt/checkpoints_grpo")
    parser.add_argument("--max_ckpt_num", type=int, default=3)
    parser.add_argument("--max_ckpt_mem", type=int, default=100000000)
    parser.add_argument("--load_checkpoint", action="store_true", default=False)
    parser.add_argument("--disable_ds_ckpt", action="store_true", default=False)
    parser.add_argument("--use_ds_universal_ckpt", action="store_true", default=False)

    parser.add_argument("--zero_stage", type=int, default=2)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=False)
    parser.add_argument("--param_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    parser.add_argument("--attn_implementation", type=str, default="flash_attention_2")

    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--rollout_batch_size", type=int, default=128)
    parser.add_argument("--micro_rollout_batch_size", type=int, default=8)
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--micro_train_batch_size", type=int, default=4)
    parser.add_argument("--prompt_max_len", type=int, default=1024)
    parser.add_argument("--generate_max_len", type=int, default=1024)
    parser.add_argument("--max_samples", type=int, default=100000000)
    parser.add_argument("--max_norm", type=float, default=1.0)
    parser.add_argument("--l2", type=float, default=0.0)
    parser.add_argument("--eps_clip", type=float, default=0.2)
    parser.add_argument("--eps_clip_low_high", type=float, nargs=2, default=None)
    parser.add_argument("--dual_clip", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--full_determinism", action="store_true", default=False)
    parser.add_argument("--n_samples_per_prompt", type=int, default=8)
    parser.add_argument("--actor_learning_rate", type=float, default=1e-6)
    parser.add_argument("--lr_warmup_ratio", type=float, default=0.03)
    parser.add_argument("--lr_scheduler", type=str, default="cosine_with_min_lr")
    parser.add_argument("--kl_target", type=float, default=None)
    parser.add_argument("--kl_horizon", type=int, default=10000)
    parser.add_argument("--init_kl_coef", type=float, default=0.01)
    parser.add_argument("--kl_estimator", type=str, default="k1", choices=["k1", "k2", "k3"])
    parser.add_argument("--use_kl_loss", action="store_true", default=False)
    parser.add_argument("--entropy_loss_coef", type=float, default=None)
    parser.add_argument("--adam_betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--reward_clip_range", type=float, nargs=2, default=(-10, 10))

    parser.add_argument("--pretrain", type=str, required=True)
    parser.add_argument("--reward_pretrain", type=str, default=None)
    parser.add_argument("--reward_fn", type=str, default=None)
    parser.add_argument("--remote_rm_url", type=str, default=None)
    parser.add_argument("--value_head_prefix", type=str, default="score")
    parser.add_argument("--normalize_reward", action="store_true", default=False)
    parser.add_argument("--ref_reward_offload", action="store_true", default=False)

    parser.add_argument("--prompt_data", type=str, required=True)
    parser.add_argument("--prompt_data_probs", type=str, default=None)
    parser.add_argument("--prompt_split", type=str, default="train")
    parser.add_argument("--eval_dataset", type=str, default=None)
    parser.add_argument("--eval_split", type=str, default="train")
    parser.add_argument("--eval_temperature", type=float, default=0.6)
    parser.add_argument("--eval_n_samples_per_prompt", type=int, default=4)
    parser.add_argument("--input_key", type=str, default="input")
    parser.add_argument("--label_key", type=str, default=None)
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--apply_chat_template", action="store_true", default=False)
    parser.add_argument("--tokenizer_chat_template", type=str, default=None)

    args = parser.parse_args()

    if args.reward_fn:
        args.remote_rm_url = args.reward_fn
    elif args.remote_rm_url and "," in args.remote_rm_url:
        args.remote_rm_url = args.remote_rm_url.split(",")

    if args.eps_clip_low_high is None:
        args.eps_clip_low_high = (args.eps_clip, args.eps_clip)

    if args.prompt_template and "{}" not in args.prompt_template:
        print("[Warning] {} not in args.prompt_template, set to None")
        args.prompt_template = None

    if args.prompt_template and "\\n" in args.prompt_template:
        print(
            "[Warning] prompt_template contains \\n characters instead of newline. "
            "You likely want to pass $'\\n' in Bash or \"`n\" in PowerShell."
        )

    assert args.n_samples_per_prompt > 1, "GRPO requires n_samples_per_prompt > 1."
    assert args.vllm_num_engines > 0, "This GRPO path requires vLLM rollout generation."
    assert args.reward_pretrain or args.remote_rm_url, "Either --reward_pretrain or --reward_fn/--remote_rm_url is required."
    if args.vllm_enable_sleep and not args.colocate_all_models:
        print("Set args.vllm_enable_sleep to False when args.colocate_all_models is disabled.")
        args.vllm_enable_sleep = False
    if args.deepspeed_enable_sleep and not args.colocate_all_models:
        print("Set args.deepspeed_enable_sleep to False when args.colocate_all_models is disabled.")
        args.deepspeed_enable_sleep = False
    if args.vllm_enable_sleep and not args.deepspeed_enable_sleep:
        print("GRPO is using deep-sleep fallback because --deepspeed_enable_sleep is disabled.")
    assert (
        args.rollout_batch_size * args.n_samples_per_prompt % args.micro_rollout_batch_size == 0
    ), "rollout_batch_size * n_samples_per_prompt must be divisible by micro_rollout_batch_size"
    assert args.train_batch_size % (args.actor_num_nodes * args.actor_num_gpus_per_node) == 0, (
        "train_batch_size must be divisible by actor world size"
    )
    assert (
        args.rollout_batch_size * args.n_samples_per_prompt % args.train_batch_size == 0
    ), "rollout_batch_size * n_samples_per_prompt must be divisible by train_batch_size"

    train(args)
