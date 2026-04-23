import math
import os
import socket
from typing import List, Optional

import deepspeed
import ray
import torch
from tqdm import tqdm
from transformers.trainer import get_scheduler

from ...models import PolicyLoss, PolicyModel, RewardModel
from ...models.utils import compute_approx_kl, masked_mean
from ..grpo_types import GRPOExperience
from ...utils import get_tokenizer
from ...utils.deepspeed import DeepspeedStrategy
from ...utils.deepspeed.deepspeed_utils import offload_deepspeed_states, reload_deepspeed_states
from ...utils.distributed_util import stateless_init_process_group, torch_dist_barrier_and_cuda_sync

from .launcher import BaseModelActor
from .utils import get_physical_gpu_id


class _ExperienceBuffer:
    def __init__(self) -> None:
        self.items: List[GRPOExperience] = []

    def append(self, experience: GRPOExperience) -> None:
        self.items.append(experience)

    def clear(self) -> None:
        self.items.clear()

    def concat(self, pad_token_id: int) -> GRPOExperience:
        return GRPOExperience.concat(self.items, pad_token_id)

    def __bool__(self) -> bool:
        return bool(self.items)


@ray.remote(num_gpus=1)
class GRPOReferenceModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain: str):
        self._setup_distributed(strategy)
        self.policy = PolicyModel(
            pretrain,
            attn_implementation=strategy.args.attn_implementation,
            param_dtype=strategy.args.param_dtype,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            temperature=strategy.args.temperature,
        )
        if strategy.args.ref_reward_offload:
            self.policy._offload = True
        self.policy = strategy.prepare(self.policy, is_rlhf=True)
        self.policy.eval()

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            log_probs = self.policy(
                sequences.to(device),
                action_mask=action_mask.to(device),
                attention_mask=attention_mask.to(device),
            )
        return log_probs.to("cpu")


@ray.remote(num_gpus=1)
class GRPORewardModelActor(BaseModelActor):
    def init_model_from_pretrained(self, strategy: DeepspeedStrategy, pretrain: str):
        self._setup_distributed(strategy)
        self.reward_model = RewardModel(
            pretrain,
            param_dtype=strategy.args.param_dtype,
            normalize_reward=strategy.args.normalize_reward,
            attn_implementation=strategy.args.attn_implementation,
            ds_config=strategy.get_ds_eval_config(offload=strategy.args.ref_reward_offload),
            value_head_prefix=strategy.args.value_head_prefix,
        )
        if strategy.args.ref_reward_offload:
            self.reward_model._offload = True
        self.reward_model = strategy.prepare(self.reward_model, is_rlhf=True)
        self.reward_model.eval()

    def forward(self, sequences: torch.LongTensor, attention_mask: torch.Tensor) -> torch.Tensor:
        device = torch.cuda.current_device()
        with torch.no_grad():
            rewards = self.reward_model(
                sequences.to(device),
                attention_mask=attention_mask.to(device),
            )
        return rewards.to("cpu")


@ray.remote(num_gpus=1)
class GRPOPolicyModelActor(BaseModelActor):
    def init_model_from_pretrained(
        self,
        strategy: DeepspeedStrategy,
        pretrain: str,
        max_steps: int,
        vllm_engines: List,
    ):
        args = strategy.args
        self._setup_distributed(strategy)
        self.vllm_engines = vllm_engines
        self.save_hf_ckpt = args.save_hf_ckpt
        self.disable_ds_ckpt = args.disable_ds_ckpt
        self.loss_fn = PolicyLoss(
            clip_eps_low=args.eps_clip_low_high[0],
            clip_eps_high=args.eps_clip_low_high[1],
            dual_clip=args.dual_clip,
        )
        self.buffer = _ExperienceBuffer()

        self.policy = PolicyModel(
            pretrain,
            attn_implementation=args.attn_implementation,
            param_dtype=args.param_dtype,
            ds_config=strategy.get_ds_train_config(is_actor=True),
            temperature=args.temperature,
        )
        self.tokenizer = get_tokenizer(pretrain, self.policy.model, "left")
        self.optimizer = strategy.create_optimizer(
            self.policy,
            lr=args.actor_learning_rate,
            betas=args.adam_betas,
            weight_decay=args.l2,
        )
        self.scheduler = get_scheduler(
            args.lr_scheduler,
            self.optimizer,
            num_warmup_steps=math.ceil(max_steps * args.lr_warmup_ratio),
            num_training_steps=max_steps,
            scheduler_specific_kwargs={"min_lr": args.actor_learning_rate * 0.1},
        )
        if args.gradient_checkpointing:
            self.policy.gradient_checkpointing_enable()

        self.policy, self.optimizer, self.scheduler = strategy.prepare(
            (self.policy, self.optimizer, self.scheduler),
            is_rlhf=True,
        )

        self.checkpoint_states = {"episode": 0, "global_step": 0, "total_consumed_prompts": 0}
        ckpt_path = os.path.join(args.ckpt_path, "_policy")
        if args.load_checkpoint and os.path.exists(ckpt_path):
            _, states = strategy.load_ckpt(self.policy.model, ckpt_path)
            self.checkpoint_states.update(states)

        backend = getattr(args, "vllm_sync_backend", "nccl")
        self.use_cuda_ipc = backend == "nccl" and getattr(args, "colocate_all_models", False)
        self._model_update_group = None
        if self.vllm_engines is not None and not self.use_cuda_ipc and torch.distributed.get_rank() == 0:
            self._init_vllm_sync_group(backend)

        if strategy.args.deepspeed_enable_sleep:
            offload_deepspeed_states(self.policy.model)
        torch_dist_barrier_and_cuda_sync()

    def _init_vllm_sync_group(self, backend: str) -> None:
        if backend != "nccl":
            raise ValueError(f"GRPO only supports --vllm_sync_backend nccl, got {backend!r}")

        master_address = ray._private.services.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind(("", 0))
            master_port = sock.getsockname()[1]

        args = self.strategy.args
        world_size = args.vllm_num_engines * args.vllm_tensor_parallel_size + 1
        refs = [
            engine.init_process_group.remote(
                master_address,
                master_port,
                i * args.vllm_tensor_parallel_size + 1,
                world_size,
                "openrlhf",
                backend=backend,
                use_ray=False,
            )
            for i, engine in enumerate(self.vllm_engines)
        ]
        self._model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            0,
            world_size,
            torch.cuda.current_device(),
        )
        ray.get(refs)

    def append(self, experience: GRPOExperience):
        self.buffer.append(experience)

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        device = torch.cuda.current_device()
        self.policy.eval()
        with torch.no_grad():
            action_log_probs = self.policy(
                sequences.to(device),
                action_mask=action_mask.to(device),
                attention_mask=attention_mask.to(device),
            )
        self.policy.train()
        return action_log_probs.to("cpu")

    def _training_step(self, batch: GRPOExperience, kl_ctl: float):
        device = torch.cuda.current_device()
        batch = batch.to_device(device)
        self.policy.train()

        action_log_probs, output = self.policy(
            batch.sequences,
            action_mask=batch.action_mask,
            attention_mask=batch.attention_mask,
            return_output=True,
            return_entropy=self.strategy.args.entropy_loss_coef is not None,
        )

        policy_loss, clip_ratio, policy_kl = self.loss_fn(
            action_log_probs,
            batch.old_action_log_probs,
            batch.advantages,
            action_mask=batch.action_mask,
        )

        if self.strategy.args.use_kl_loss and batch.base_action_log_probs is not None:
            kl = compute_approx_kl(action_log_probs, batch.base_action_log_probs, self.strategy.args.kl_estimator)
            kl_loss = masked_mean(kl, batch.action_mask)
            logprobs_diff = masked_mean(action_log_probs.float() - batch.base_action_log_probs.float(), batch.action_mask)
        else:
            kl_loss = torch.tensor(0.0, device=device)
            logprobs_diff = torch.tensor(0.0, device=device)

        loss = policy_loss + kl_ctl * kl_loss
        entropy_loss = None
        if self.strategy.args.entropy_loss_coef is not None:
            entropy_loss = masked_mean(output.entropy[:, -batch.action_mask.shape[1] :], batch.action_mask)
            if self.strategy.args.entropy_loss_coef != 0:
                loss -= self.strategy.args.entropy_loss_coef * entropy_loss

        self.strategy.backward(loss, self.policy, self.optimizer)
        self.strategy.optimizer_step(self.optimizer, self.policy, self.scheduler, name="policy")

        status = {
            "policy_loss": policy_loss.detach().item(),
            "clip_ratio": clip_ratio.detach().item(),
            "policy_kl": policy_kl.detach().item(),
            "kl": kl_loss.detach().item(),
            "logprobs_diff": logprobs_diff.detach().item(),
            "actor_lr": self.scheduler.get_last_lr()[0],
        }
        if entropy_loss is not None:
            status["entropy_loss"] = entropy_loss.detach().item()
        for key, value in batch.info.items():
            if isinstance(value, torch.Tensor):
                status[key] = value.float().mean().item()
        return status

    def fit(self, kl_ctl: float = 0.0):
        if not self.buffer:
            return {}

        torch.cuda.empty_cache()
        experience = self.buffer.concat(self.tokenizer.pad_token_id)
        self.buffer.clear()
        num_samples = len(experience)
        status_list = []

        for epoch in range(self.strategy.args.max_epochs):
            indices = torch.randperm(num_samples)
            pbar = tqdm(
                range(0, num_samples, self.strategy.args.micro_train_batch_size),
                desc=f"Train epoch [{epoch + 1}/{self.strategy.args.max_epochs}]",
                disable=not self.strategy.is_rank_0(),
            )
            for start in pbar:
                batch = experience.select(indices[start : start + self.strategy.args.micro_train_batch_size])
                status = self.strategy.all_reduce(self._training_step(batch, kl_ctl))
                status_list.append(status)
                pbar.set_postfix(
                    {
                        "act_loss": status["policy_loss"],
                        "reward": status["reward"],
                        "return": status["return"],
                        "gen_len": status["response_length"],
                        "tot_len": status["total_length"],
                        "kl": status["kl"],
                        "act_lr": status["actor_lr"],
                    }
                )

        mean_status = status_list[0]
        for status in status_list[1:]:
            for key in mean_status:
                mean_status[key] += status[key]
        for key in mean_status:
            mean_status[key] /= len(status_list)
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return mean_status

    def broadcast_to_vllm(self):
        if self.strategy.args.enable_prefix_caching and torch.distributed.get_rank() == 0:
            ray.get([engine.reset_prefix_cache.remote() for engine in self.vllm_engines])

        torch.cuda.empty_cache()
        model = self.policy.model.module
        num_params = len(list(model.named_parameters()))

        def _broadcast_param(name, param):
            shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
            refs = [engine.update_weight.remote(name, dtype=param.dtype, shape=shape) for engine in self.vllm_engines]
            self._model_update_group.broadcast(param.data, src=0, stream=torch.cuda.current_stream())
            ray.get(refs)

        def _handle_cuda_ipc(name, param, count):
            from torch.multiprocessing.reductions import reduce_tensor

            weight = param.data.clone()
            ipc_handle = reduce_tensor(weight)

            ipc_handle = {get_physical_gpu_id(): ipc_handle}
            ipc_handle_list = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(ipc_handle_list, ipc_handle)

            if torch.distributed.get_rank() == 0:
                ipc_handles = {}
                for d in ipc_handle_list:
                    ipc_handles.update(d)

                shape = param.shape if self.strategy.args.zero_stage != 3 else param.ds_shape
                refs = [
                    engine.update_weight_cuda_ipc.remote(
                        name,
                        dtype=param.dtype,
                        shape=shape,
                        ipc_handles=ipc_handles,
                        empty_cache=count == num_params,
                    )
                    for engine in self.vllm_engines
                ]
                ray.get(refs)
            torch_dist_barrier_and_cuda_sync()

        for count, (name, param) in enumerate(model.named_parameters(), start=1):
            with deepspeed.zero.GatheredParameters([param], enabled=self.strategy.args.zero_stage == 3):
                if torch.distributed.get_rank() == 0:
                    if self.use_cuda_ipc:
                        _handle_cuda_ipc(name, param, count)
                    else:
                        _broadcast_param(name, param)
                elif self.use_cuda_ipc:
                    _handle_cuda_ipc(name, param, count)
        torch_dist_barrier_and_cuda_sync()

    def get_checkpoint_states(self):
        return self.checkpoint_states

    def reload_states(self):
        reload_deepspeed_states(self.policy.model)

    def offload_states(self):
        offload_deepspeed_states(self.policy.model)

    def save_checkpoint(self, tag: str, client_states):
        if not self.disable_ds_ckpt:
            self.strategy.save_ckpt(
                self.policy.model,
                os.path.join(self.strategy.args.ckpt_path, "_policy"),
                tag,
                self.strategy.args.max_ckpt_num,
                self.strategy.args.max_ckpt_mem,
                client_states,
            )
        if self.save_hf_ckpt:
            save_path = os.path.join(self.strategy.args.ckpt_path, f"{tag}_hf")
            self.strategy.save_model(self.policy, self.tokenizer, save_path)
        torch_dist_barrier_and_cuda_sync()

    def save_model(self):
        self.strategy.save_model(self.policy, self.tokenizer, self.strategy.args.save_path)
