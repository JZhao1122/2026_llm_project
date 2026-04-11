import os
from abc import ABC

import torch
from torch.optim import Optimizer
from tqdm import tqdm

from models import SFTLoss
from ..utils.distributed_sampler import DistributedSampler
from ..utils.logging_utils import JsonlLogger


class SFTTrainer(ABC):
    def __init__(
        self,
        model,
        strategy,
        optim: Optimizer,
        train_dataloader,
        eval_dataloader,
        scheduler,
        batch_size: int = 1,
        max_epochs: int = 2,
        tokenizer=None,
        save_hf_ckpt: bool = False,
        disable_ds_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.strategy = strategy
        self.epochs = max_epochs
        self.batch_size = batch_size
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.scheduler = scheduler
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optim
        self.args = strategy.args
        self.save_hf_ckpt = save_hf_ckpt
        self.disable_ds_ckpt = disable_ds_ckpt
        self.loss_fn = SFTLoss()
        self.jsonl_logger = JsonlLogger(os.path.join(self.args.save_path, "metrics.jsonl")) if strategy.is_rank_0() else None

    def fit(self, args, consumed_samples=0, num_update_steps_per_epoch=None):
        if args.eval_steps == -1:
            args.eval_steps = num_update_steps_per_epoch
        if args.save_steps == -1:
            args.save_steps = float("inf")

        step = consumed_samples // args.train_batch_size * self.strategy.accumulated_gradient + 1
        start_epoch = consumed_samples // args.train_batch_size // num_update_steps_per_epoch
        consumed_samples = consumed_samples % (num_update_steps_per_epoch * args.train_batch_size)

        epoch_bar = tqdm(range(start_epoch, self.epochs), desc="Train epoch", disable=not self.strategy.is_rank_0())
        loss_sum = 0.0
        for epoch in range(start_epoch, self.epochs):
            if isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(
                    epoch, consumed_samples=0 if epoch > start_epoch else consumed_samples
                )

            step_bar = tqdm(
                range(self.train_dataloader.__len__()),
                desc=f"Train step of epoch {epoch}",
                disable=not self.strategy.is_rank_0(),
            )

            self.model.train()
            for inputs, attention_masks, loss_masks in self.train_dataloader:
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                loss_mask = loss_masks.to(torch.cuda.current_device()).squeeze(1)

                per_token_log_probs = self.model(inputs, attention_mask=attention_mask)
                loss = self.loss_fn(per_token_log_probs, loss_mask[:, :-1])
                self.strategy.backward(loss, self.model, self.optimizer)
                self.strategy.optimizer_step(self.optimizer, self.model, self.scheduler)

                loss_sum += loss.item()
                logs_dict = self.strategy.all_reduce(
                    {
                        "gpt_loss": loss.item(),
                        "lr": self.scheduler.get_last_lr()[0],
                    }
                )
                step_bar.set_postfix(logs_dict)
                step_bar.update()

                if step % self.strategy.accumulated_gradient == 0:
                    logs_dict["loss_mean"] = loss_sum / self.strategy.accumulated_gradient
                    loss_sum = 0.0
                    global_step = step // self.strategy.accumulated_gradient
                    client_states = {"consumed_samples": global_step * args.train_batch_size}
                    self.save_logs_and_checkpoints(args, global_step, logs_dict, client_states)

                step += 1

            epoch_bar.update()

        if self.jsonl_logger is not None:
            self.jsonl_logger.close()

    def save_logs_and_checkpoints(self, args, global_step, logs_dict=None, client_states=None):
        logs_dict = logs_dict or {}
        if global_step % args.logging_steps == 0 and self.jsonl_logger is not None and self.strategy.is_rank_0():
            self.jsonl_logger.log_train(global_step, logs_dict)

        if global_step % args.eval_steps == 0 and self.eval_dataloader is not None and len(self.eval_dataloader) > 0:
            self.evaluate(self.eval_dataloader, global_step)

        client_states = client_states or {}
        if global_step % args.save_steps == 0:
            tag = f"global_step{global_step}"
            if not self.disable_ds_ckpt:
                self.strategy.save_ckpt(
                    self.model.model, args.ckpt_path, tag, args.max_ckpt_num, args.max_ckpt_mem, client_states
                )
            if self.save_hf_ckpt:
                save_path = os.path.join(args.ckpt_path, f"{tag}_hf")
                self.strategy.save_model(self.model, self.tokenizer, save_path)

    def evaluate(self, eval_dataloader, steps=0):
        self.model.eval()
        with torch.no_grad():
            loss_sum = 0.0
            times = 0
            step_bar = tqdm(
                range(eval_dataloader.__len__()),
                desc=f"Eval stage of steps {steps}",
                disable=not self.strategy.is_rank_0(),
            )

            for inputs, attention_masks, loss_masks in eval_dataloader:
                inputs = inputs.to(torch.cuda.current_device()).squeeze(1)
                attention_mask = attention_masks.to(torch.cuda.current_device()).squeeze(1)
                loss_mask = loss_masks.to(torch.cuda.current_device()).squeeze(1)

                per_token_log_probs = self.model(inputs, attention_mask=attention_mask)
                loss = self.loss_fn(per_token_log_probs, loss_mask[:, :-1])

                times += 1
                loss_sum += loss.item()
                logs = self.strategy.all_reduce({"eval_gpt_loss": loss_sum / times})
                step_bar.update()
                step_bar.set_postfix(logs)

            if self.strategy.is_rank_0() and self.jsonl_logger is not None:
                self.jsonl_logger.log_eval(steps, logs)
        self.model.train()
