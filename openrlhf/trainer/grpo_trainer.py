import math
import os
from datetime import timedelta
from typing import Dict, List, Optional, Tuple

import ray
import torch
from tqdm import tqdm
from vllm import SamplingParams

from openrlhf.datasets import PromptDataset
from openrlhf.datasets.utils import blending_datasets
from openrlhf.models.utils import compute_approx_kl, compute_reward, masked_mean
from openrlhf.trainer.grpo_types import GRPOExperience
from openrlhf.trainer.ray import batch_vllm_engine_call
from openrlhf.utils.logging_utils import JsonlLogger, init_logger

logger = init_logger(__name__)


# ============================================================
# KL Controllers
# ============================================================


class AdaptiveKLController:
    def __init__(self, init_kl_coef: float, target: float, horizon: int):
        self.value = init_kl_coef
        self.target = target
        self.horizon = horizon

    def update(self, current: float, n_steps: int) -> None:
        proportional_error = min(max(current / self.target - 1, -0.2), 0.2)
        self.value *= 1 + proportional_error * n_steps / self.horizon


class FixedKLController:
    def __init__(self, kl_coef: float):
        self.value = kl_coef

    def update(self, current: float, n_steps: int) -> None:
        return None


# ============================================================
# Dataset preparation
# ============================================================


def prepare_datasets(args, strategy, tokenizer):
    train_data = blending_datasets(
        args.prompt_data,
        args.prompt_data_probs,
        args.seed,
        max_count=args.max_samples,
        dataset_split=args.prompt_split,
        is_rank_0=strategy.is_rank_0(),
    )
    train_data = train_data.select(range(min(args.max_samples, len(train_data))))
    train_dataset = PromptDataset(
        train_data,
        tokenizer,
        input_key=args.input_key,
        label_key=args.label_key,
        prompt_template=args.prompt_template,
        apply_chat_template=args.apply_chat_template,
        tokenizer_chat_template=args.tokenizer_chat_template,
        show_progress=strategy.is_rank_0(),
    )
    train_dataloader = strategy.setup_dataloader(
        train_dataset,
        args.rollout_batch_size,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
        drop_last=True,
    )

    eval_dataloader = None
    if args.eval_dataset:
        eval_data = blending_datasets(
            args.eval_dataset,
            None,
            dataset_split=args.eval_split,
            is_rank_0=strategy.is_rank_0(),
        )
        eval_data = eval_data.select(range(min(args.max_samples, len(eval_data))))
        eval_dataset = PromptDataset(
            eval_data,
            tokenizer,
            input_key=args.input_key,
            label_key=args.label_key,
            prompt_template=args.prompt_template,
            apply_chat_template=args.apply_chat_template,
            tokenizer_chat_template=args.tokenizer_chat_template,
            show_progress=strategy.is_rank_0(),
        )
        eval_dataloader = strategy.setup_dataloader(
            eval_dataset,
            args.rollout_batch_size,
            shuffle=False,
            collate_fn=eval_dataset.collate_fn,
            drop_last=False,
        )

    updates_per_rollout = args.rollout_batch_size * args.n_samples_per_prompt // args.train_batch_size
    max_steps = len(train_dataloader) * args.num_episodes * args.max_epochs * updates_per_rollout
    return train_dataloader, eval_dataloader, max_steps


# ============================================================
# Rollout generation (vLLM)
# ============================================================


class VLLMRolloutGenerator:
    def __init__(self, strategy, tokenizer, vllm_engines):
        self.args = strategy.args
        self.tokenizer = tokenizer
        self.vllm_engines = vllm_engines

    def _response_to_experience(self, response, max_length: int) -> GRPOExperience:
        sequences = torch.tensor(response["observation_tokens"][:max_length], dtype=torch.long)
        attention_mask = torch.ones_like(sequences, dtype=torch.long)

        action_mask = torch.zeros(max(sequences.size(0) - 1, 0), dtype=torch.bool)
        for start, end in response["action_ranges"]:
            start = max(start - 1, 0)
            end = min(end - 1, action_mask.size(0))
            if start < end:
                action_mask[start:end] = True

        if action_mask.any():
            response_length = float(action_mask.sum().item())
        else:
            response_length = 0.0
        total_length = float(attention_mask.sum().item())

        reward = response.get("reward")
        score = response.get("scores", reward)
        info = {
            "response_length": torch.tensor([response_length], dtype=torch.float32),
            "total_length": torch.tensor([total_length], dtype=torch.float32),
            "truncated": torch.tensor([float(response.get("truncated", False))], dtype=torch.float32),
        }
        if reward is not None:
            info["reward"] = torch.tensor([float(reward)], dtype=torch.float32)
        if score is not None:
            info["score"] = torch.tensor([float(score)], dtype=torch.float32)
        for key, value in (response.get("extra_logs") or {}).items():
            if isinstance(value, torch.Tensor):
                value = value.reshape(-1)[0].item()
            info[key] = torch.tensor([float(value)], dtype=torch.float32)

        rewards = None if reward is None else torch.tensor([float(reward)], dtype=torch.float32)
        return GRPOExperience(
            sequences=sequences.unsqueeze(0),
            attention_mask=attention_mask.unsqueeze(0),
            action_mask=action_mask.unsqueeze(0),
            rewards=rewards,
            prompts=[response["prompt"]],
            labels=[response["label"]],
            info=info,
        )

    def generate(self, prompts: List[str], labels: List[str], *, temperature: float, num_samples: int) -> List[GRPOExperience]:
        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "wake_up")

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=self.args.top_p,
            max_tokens=self.args.generate_max_len,
            min_tokens=1,
            skip_special_tokens=False,
        )
        max_length = self.args.prompt_max_len + self.args.generate_max_len

        refs = []
        for index, (prompt, label) in enumerate(zip(prompts, labels)):
            engine = self.vllm_engines[index % len(self.vllm_engines)]
            refs.append(
                engine.generate_responses.remote(
                    prompt=prompt,
                    label=label,
                    sampling_params=sampling_params,
                    max_length=max_length,
                    hf_tokenizer=self.tokenizer,
                    num_samples=num_samples,
                )
            )

        responses_per_prompt = ray.get(refs)

        if self.args.vllm_enable_sleep:
            batch_vllm_engine_call(self.vllm_engines, "sleep")

        samples = []
        for responses in responses_per_prompt:
            for response in responses:
                samples.append(self._response_to_experience(response, max_length))
        return samples


# ============================================================
# GRPO Trainer
# ============================================================


class RayGRPOTrainer:
    """
    Trainer implementing Group Relative Policy Optimization (GRPO).

    GRPO generates multiple responses per prompt, computes group-relative
    advantages by normalizing rewards within each prompt group, and updates
    the policy using a clipped surrogate objective with optional KL penalty.

    The training loop follows these steps for each batch of prompts:
        1. Generate rollouts   -- sample N responses per prompt via vLLM
        2. Compute log probs   -- get pi(a|s) from both actor and reference
        3. Compute rewards     -- score responses (rule-based or reward model)
        4. Compute advantages  -- GRPO group normalization of rewards
        5. Compute KL/returns  -- KL penalty + discounted return estimation
        6. Policy update       -- clipped PPO-style gradient steps
        7. Sync weights        -- broadcast updated params to vLLM engines
    """

    def __init__(
        self,
        strategy,
        actor_model_group,
        reference_model_group,
        reward_model_group,
        vllm_engines,
        tokenizer,
        train_dataloader,
        eval_dataloader,
    ):
        self.strategy = strategy
        self.args = strategy.args
        self.actor_model_group = actor_model_group
        self.reference_model_group = reference_model_group
        self.reward_model_group = reward_model_group
        self.rollout_generator = VLLMRolloutGenerator(strategy, tokenizer, vllm_engines)
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.jsonl_logger = JsonlLogger(os.path.join(self.args.save_path, "metrics.jsonl")) if strategy.is_rank_0() else None

        if self.args.kl_target:
            self.kl_ctl = AdaptiveKLController(self.args.init_kl_coef, self.args.kl_target, self.args.kl_horizon)
        else:
            self.kl_ctl = FixedKLController(self.args.init_kl_coef)

        self.checkpoint_states = {"episode": 0, "global_step": 0, "total_consumed_prompts": 0}
        if self.args.load_checkpoint:
            states = ray.get(self.actor_model_group.async_run_method(method_name="get_checkpoint_states"))[0]
            self.checkpoint_states.update(states)

    # --------------------------------------------------------
    # Micro-batch helpers
    # --------------------------------------------------------

    def _num_sample_batches(self, num_samples: int) -> int:
        return math.ceil(num_samples / self.args.micro_rollout_batch_size)

    def _validate_micro_rollout_batches(self, num_samples: int) -> None:
        if num_samples % self.args.micro_rollout_batch_size != 0:
            raise ValueError(
                "num_samples must be divisible by micro_rollout_batch_size so Ray workers receive identical batch counts"
            )
        num_batches = self._num_sample_batches(num_samples)
        actor_world_size = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        if num_batches < actor_world_size or num_batches % actor_world_size != 0:
            raise ValueError(
                f"Need sample batch count divisible by actor workers ({actor_world_size}), got {num_batches}. "
                "Decrease --micro_rollout_batch_size or increase rollout_batch_size."
            )
        if self.reference_model_group is not None:
            ref_world_size = self.args.ref_num_nodes * self.args.ref_num_gpus_per_node
            if num_batches < ref_world_size or num_batches % ref_world_size != 0:
                raise ValueError(
                    f"Need sample batch count divisible by reference workers ({ref_world_size}), got {num_batches}."
                )
        if self.reward_model_group is not None:
            reward_world_size = self.args.reward_num_nodes * self.args.reward_num_gpus_per_node
            if num_batches < reward_world_size or num_batches % reward_world_size != 0:
                raise ValueError(
                    f"Need sample batch count divisible by reward workers ({reward_world_size}), got {num_batches}."
                )

    def _split_into_micro_batches(self, samples: List[GRPOExperience]) -> List[GRPOExperience]:
        self._validate_micro_rollout_batches(len(samples))
        return [
            GRPOExperience.concat(samples[i : i + self.args.micro_rollout_batch_size], self.tokenizer.pad_token_id)
            for i in range(0, len(samples), self.args.micro_rollout_batch_size)
        ]

    def _flatten_refs(self, refs):
        return sum(ray.get(refs), [])

    # --------------------------------------------------------
    # Step 1: Generate rollouts
    # --------------------------------------------------------

    def _generate_rollouts(self, prompts: List[str], labels: List[str]) -> List[GRPOExperience]:
        """Sample N responses per prompt using vLLM."""
        return self.rollout_generator.generate(
            prompts,
            labels,
            temperature=self.args.temperature,
            num_samples=self.args.n_samples_per_prompt,
        )

    # --------------------------------------------------------
    # Step 2: Compute log probabilities
    # --------------------------------------------------------

    def _compute_action_log_probs(self, batches: List[GRPOExperience]) -> List[torch.Tensor]:
        """Compute log pi_actor(a|s) for each micro-batch."""
        return self._flatten_refs(
            self.actor_model_group.async_run_method_batch(
                method_name="forward",
                sequences=[b.sequences for b in batches],
                action_mask=[b.action_mask for b in batches],
                attention_mask=[b.attention_mask for b in batches],
            )
        )

    def _compute_reference_log_probs(self, batches: List[GRPOExperience]) -> List[Optional[torch.Tensor]]:
        """Compute log pi_ref(a|s) for each micro-batch. Returns Nones if no reference model."""
        if self.reference_model_group is None:
            return [None] * len(batches)
        return self._flatten_refs(
            self.reference_model_group.async_run_method_batch(
                method_name="forward",
                sequences=[b.sequences for b in batches],
                action_mask=[b.action_mask for b in batches],
                attention_mask=[b.attention_mask for b in batches],
            )
        )

    # --------------------------------------------------------
    # Step 3: Compute rewards
    # --------------------------------------------------------

    def _score_with_reward_model(self, batches: List[GRPOExperience]) -> None:
        """Score responses using a learned reward model (mutates batches in-place)."""
        if self.reward_model_group is None:
            return
        reward_refs = self.reward_model_group.async_run_method_batch(
            method_name="forward",
            sequences=[b.sequences for b in batches],
            attention_mask=[b.attention_mask for b in batches],
        )
        for batch, reward in zip(batches, self._flatten_refs(reward_refs)):
            batch.rewards = reward.float()
            batch.info["reward"] = reward.float()
            batch.info["score"] = reward.float()

    # --------------------------------------------------------
    # Step 4: Compute GRPO group-relative advantages
    # --------------------------------------------------------

    def _compute_grpo_advantages(
        self, batches: List[GRPOExperience]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize rewards within each prompt group (the core GRPO idea).

        For each prompt with N sampled responses, the advantage is:
            A_i = (r_i - mean(r_1..N)) / (std(r_1..N) + eps)

        Returns:
            normalized_rewards: per-sample normalized reward (flat tensor)
            group_reward_std:   per-sample group std for logging (flat tensor)
        """
        flat_rewards = torch.cat([b.rewards.float().reshape(-1) for b in batches], dim=0)
        group_rewards = flat_rewards.view(-1, self.args.n_samples_per_prompt)

        mean = group_rewards.mean(dim=-1, keepdim=True)
        std = group_rewards.std(dim=-1, keepdim=True)
        normalized_rewards = (group_rewards - mean) / (std + 1e-9)

        group_reward_std = std.repeat(1, self.args.n_samples_per_prompt).reshape(-1)
        return normalized_rewards.reshape(-1), group_reward_std

    # --------------------------------------------------------
    # Step 5: Compute KL penalty and returns
    # --------------------------------------------------------

    def _compute_discounted_returns(self, token_rewards: torch.Tensor, action_mask: torch.Tensor) -> torch.Tensor:
        """Compute discounted cumulative returns from per-token rewards."""
        returns = torch.zeros_like(token_rewards)
        running = torch.zeros(token_rewards.size(0), device=token_rewards.device)
        masked_rewards = token_rewards * action_mask
        for t in reversed(range(token_rewards.size(1))):
            running = masked_rewards[:, t] + self.args.gamma * running
            returns[:, t] = running
        return returns

    def _compute_kl_and_returns(
        self,
        batches: List[GRPOExperience],
        action_log_probs: List[torch.Tensor],
        ref_log_probs: List[Optional[torch.Tensor]],
        normalized_rewards: torch.Tensor,
        group_reward_std: torch.Tensor,
    ) -> None:
        """
        For each micro-batch, compute:
            - KL(pi_actor || pi_ref) as a per-token penalty
            - Token-level rewards = group-normalized advantage at EOS + KL penalty
            - Discounted returns from token-level rewards

        Mutates batches in-place to set old_action_log_probs, returns, advantages, etc.
        """
        offset = 0
        for batch, old_log_probs, base_log_probs in zip(batches, action_log_probs, ref_log_probs):
            batch_size = len(batch)
            reward_slice = normalized_rewards[offset : offset + batch_size].to(old_log_probs.device)
            std_slice = group_reward_std[offset : offset + batch_size].to(old_log_probs.device)
            offset += batch_size

            # KL divergence between current policy and reference policy
            if base_log_probs is not None and not self.args.use_kl_loss:
                kl = compute_approx_kl(old_log_probs, base_log_probs, kl_estimator=self.args.kl_estimator)
            else:
                kl = torch.zeros_like(old_log_probs)

            # Token-level rewards: place scalar advantage at EOS, subtract KL penalty elsewhere
            token_rewards = compute_reward(
                reward_slice,
                self.kl_ctl.value,
                kl,
                action_mask=batch.action_mask,
                reward_clip_range=self.args.reward_clip_range,
            )

            # Populate batch fields for the policy update
            batch.old_action_log_probs = old_log_probs
            batch.base_action_log_probs = base_log_probs if self.args.use_kl_loss else None
            batch.returns = self._compute_discounted_returns(token_rewards, batch.action_mask)
            batch.advantages = batch.returns.clone()

            # Logging info
            batch.info["return"] = token_rewards.sum(dim=-1).detach().cpu()
            batch.info["group_reward_std"] = std_slice.detach().cpu()
            if base_log_probs is not None and not self.args.use_kl_loss:
                batch.info["kl"] = masked_mean(kl, batch.action_mask, dim=-1).detach().cpu()

    # --------------------------------------------------------
    # Step 6: Policy gradient update
    # --------------------------------------------------------

    def _policy_update(self, batches: List[GRPOExperience]) -> Dict[str, float]:
        """Push experience to actor workers and run clipped PPO-style gradient steps."""
        ray.get(self.actor_model_group.async_run_method_batch(method_name="append", experience=batches))
        status_list = ray.get(self.actor_model_group.async_run_method(method_name="fit", kl_ctl=self.kl_ctl.value))
        return status_list[0]

    # --------------------------------------------------------
    # Step 7: Sync weights to vLLM
    # --------------------------------------------------------

    def _sync_weights_to_vllm(self) -> None:
        """Broadcast updated actor weights to all vLLM inference engines."""
        ray.get(self.actor_model_group.async_run_method(method_name="broadcast_to_vllm"))

    # --------------------------------------------------------
    # Logging and checkpointing
    # --------------------------------------------------------

    def _save_logs_and_checkpoints(self, global_step: int, status: Dict[str, float], client_states: Dict[str, int]) -> None:
        if global_step % self.args.logging_steps == 0 and self.jsonl_logger is not None:
            self.jsonl_logger.log_train(global_step, status)

        if global_step % self.args.save_steps == 0:
            tag = f"global_step{global_step}"
            ray.get(
                self.actor_model_group.async_run_method(
                    method_name="save_checkpoint",
                    tag=tag,
                    client_states=client_states,
                )
            )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    def _score_samples(self, samples: List[GRPOExperience]) -> torch.Tensor:
        if samples[0].rewards is not None:
            return torch.cat([sample.rewards for sample in samples], dim=0)
        batches = self._split_into_micro_batches(samples)
        reward_refs = self.reward_model_group.async_run_method_batch(
            method_name="forward",
            sequences=[b.sequences for b in batches],
            attention_mask=[b.attention_mask for b in batches],
        )
        return torch.cat([r.float().reshape(-1) for r in self._flatten_refs(reward_refs)], dim=0)

    def evaluate(self, global_step: int) -> None:
        if self.eval_dataloader is None:
            return

        pass_at_1 = 0.0
        pass_at_k = 0.0
        total = 0
        k = self.args.eval_n_samples_per_prompt

        for _, prompts, labels in self.eval_dataloader:
            samples = self.rollout_generator.generate(
                prompts,
                labels,
                temperature=self.args.eval_temperature,
                num_samples=k,
            )
            rewards = self._score_samples(samples).view(len(prompts), k)
            pass_at_k += rewards.max(dim=-1).values.sum().item()
            pass_at_1 += rewards.mean(dim=-1).sum().item()
            total += len(prompts)

        if total == 0:
            return

        logs = {f"eval_pass{k}": pass_at_k / total, "eval_pass1": pass_at_1 / total}
        if self.jsonl_logger is not None:
            self.jsonl_logger.log_eval(global_step, logs)
        logger.info(f"Evaluation step {global_step}: {logs}")

    # --------------------------------------------------------
    # Main training loop
    # --------------------------------------------------------

    def fit(self):
        if self.args.eval_steps == -1:
            self.args.eval_steps = float("inf")
        if self.args.save_steps == -1:
            self.args.save_steps = float("inf")

        global_step = self.checkpoint_states["global_step"]
        total_consumed_prompts = self.checkpoint_states["total_consumed_prompts"]
        start_episode = self.checkpoint_states["episode"]

        for episode in range(start_episode, self.args.num_episodes):
            pbar = tqdm(
                self.train_dataloader,
                desc=f"Episode [{episode + 1}/{self.args.num_episodes}]",
                disable=not self.strategy.is_rank_0(),
            )
            for _, prompts, labels in pbar:

                # Step 1: Generate N rollouts per prompt using vLLM
                samples = self._generate_rollouts(prompts, labels)
                batches = self._split_into_micro_batches(samples)

                # Step 2: Compute log probs under current policy and reference policy
                action_log_probs = self._compute_action_log_probs(batches)
                ref_log_probs = self._compute_reference_log_probs(batches)

                # Step 3: Score responses with reward model (if not already scored by rules)
                self._score_with_reward_model(batches)

                # Step 4: Compute GRPO group-relative advantages
                normalized_rewards, group_reward_std = self._compute_grpo_advantages(batches)

                # Step 5: Compute KL penalty and discounted returns
                self._compute_kl_and_returns(
                    batches, action_log_probs, ref_log_probs, normalized_rewards, group_reward_std
                )

                # Step 6: Run policy gradient update (clipped PPO-style)
                status = self._policy_update(batches)

                # Step 7: Broadcast updated weights to vLLM engines
                self._sync_weights_to_vllm()

                # Update adaptive KL controller
                if "kl" in status:
                    self.kl_ctl.update(status["kl"], self.args.rollout_batch_size * self.args.n_samples_per_prompt)

                # Progress tracking
                global_step += 1
                total_consumed_prompts += len(prompts)
                logger.info(f"Global step {global_step}: {status}")
                pbar.set_postfix(
                    {
                        "act_loss": status["policy_loss"],
                        "reward": status["reward"],
                        "return": status["return"],
                        "kl": status["kl"],
                    }
                )

                # Logging, checkpointing, and evaluation
                client_states = {
                    "episode": episode,
                    "global_step": global_step,
                    "total_consumed_prompts": total_consumed_prompts,
                }
                self._save_logs_and_checkpoints(global_step, status, client_states)

                if global_step % self.args.eval_steps == 0:
                    self.evaluate(global_step)

        if self.jsonl_logger is not None:
            self.jsonl_logger.close()
