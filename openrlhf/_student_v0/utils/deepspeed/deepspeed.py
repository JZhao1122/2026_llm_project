import gc
import os
import shutil
from abc import ABC
from datetime import timedelta
from typing import List, Tuple, Union

import deepspeed
import torch
import torch.nn as nn
import torch.optim as optim
import transformers
from deepspeed.ops.adam import FusedAdam
from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.optim import Optimizer
from torchdata.stateful_dataloader import StatefulDataLoader

from ..distributed_sampler import DistributedSampler
from ..distributed_util import torch_dist_barrier_and_cuda_sync
from .deepspeed_utils import get_eval_ds_config, get_optimizer_grouped_parameters, get_train_ds_config

ModelOptimPair = Tuple[nn.Module, Optimizer]
ModelOrModelOptimPair = Union[nn.Module, ModelOptimPair]


class DeepspeedStrategy(ABC):
    """Distributed training strategy used by the trimmed SFT and GRPO paths."""

    def __init__(
        self,
        seed: int = 42,
        max_norm: float = 0.0,
        micro_train_batch_size=1,
        train_batch_size=1,
        zero_stage=2,
        args=None,
    ) -> None:
        super().__init__()

        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.micro_train_batch_size = micro_train_batch_size
        self.param_dtype = args.param_dtype
        self.seed = seed
        self.max_norm = max_norm
        self.use_ds_universal_ckpt = getattr(args, "use_ds_universal_ckpt", False)
        self.is_rlhf = False

    def _is_wrapper(self, model) -> bool:
        return hasattr(model, "model")

    def _extract_model(self, model):
        return model.model if self._is_wrapper(model) else model

    def setup_distributed(self, timeout=timedelta(minutes=60)) -> None:
        transformers.set_seed(self.seed)

        if self.args.local_rank != -1:
            os.environ["LOCAL_RANK"] = str(self.args.local_rank)

        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank != -1:
            torch.cuda.set_device(local_rank)

        deepspeed.init_distributed(timeout=timeout)

        self.world_size = dist.get_world_size()
        self.ds_device_mesh = init_device_mesh("cuda", (self.world_size,), mesh_dim_names=("dp",))
        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size // self.world_size
        self.accumulated_gradient = max(self.accumulated_gradient, 1)

    def create_optimizer(self, model, **kwargs) -> Optimizer:
        model = self._extract_model(model)
        optim_params = get_optimizer_grouped_parameters(model, kwargs["weight_decay"])
        return FusedAdam(optim_params, **kwargs)

    def backward(self, loss: torch.Tensor, model: nn.Module, optimizer: optim.Optimizer, **kwargs) -> None:
        self._extract_model(model).backward(loss)

    def optimizer_step(
        self,
        optimizer: optim.Optimizer,
        model: nn.Module,
        scheduler,
        name="model",
        **kwargs,
    ) -> None:
        self._extract_model(model).step()

    def setup_dataloader(
        self,
        replay_buffer,
        batch_size: int,
        pin_memory: bool = False,
        shuffle=True,
        collate_fn=None,
        drop_last=True,
        sampler=None,
        consumed_samples=0,
    ):
        if sampler is None and dist.is_initialized():
            dp_group = self.ds_device_mesh["dp"].get_group()
            num_replicas = dist.get_world_size(group=dp_group)
            rank = dist.get_rank(group=dp_group)
            sampler = DistributedSampler(
                replay_buffer,
                num_replicas=num_replicas,
                rank=rank,
                shuffle=shuffle,
                seed=self.seed,
                drop_last=drop_last,
                consumed_samples=consumed_samples,
            )

        return StatefulDataLoader(
            replay_buffer,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=drop_last,
            shuffle=shuffle if sampler is None else False,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )

    def _unwrap_model(self, model) -> nn.Module:
        if self._is_wrapper(model):
            return self._unwrap_model(model.model)
        if hasattr(model, "module"):
            return model.module
        return model

    def prepare(
        self, *models_or_model_optim_pairs: ModelOrModelOptimPair, is_rlhf=False
    ) -> Union[List[ModelOrModelOptimPair], ModelOrModelOptimPair]:
        ret = []
        self.is_rlhf = is_rlhf
        for arg in models_or_model_optim_pairs:
            if isinstance(arg, tuple):
                assert len(arg) == 3, f'Expect (model, optimizer, scheduler) pair, got a tuple with size "{len(arg)}"'
                if arg[0] is not None:
                    ret.append(self._ds_init_train_model(*arg))
                else:
                    ret.append((None, None, None))
            else:
                ret.append(self._ds_init_eval_model(arg))

        return ret[0] if len(ret) == 1 else ret

    def _ds_init_train_model(self, model, optim, scheduler):
        is_wrapper = self._is_wrapper(model)
        engine, optim, _, scheduler = deepspeed.initialize(
            model=self._extract_model(model),
            optimizer=optim,
            lr_scheduler=scheduler,
            config=self.get_ds_train_config(is_actor=is_wrapper),
            args={"local_rank": int(os.environ.get("LOCAL_RANK", "-1"))},
            dist_init_required=True,
        )
        if is_wrapper:
            model.model = engine
        else:
            model = engine
        return model, optim, scheduler

    def get_ds_train_config(self, is_actor):
        ds_config = get_train_ds_config(
            offload=False,
            stage=self.stage,
            param_dtype=self.param_dtype,
            max_norm=self.max_norm,
            use_ds_universal_ckpt=self.use_ds_universal_ckpt,
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        ds_config["train_batch_size"] = self.train_batch_size
        return ds_config

    def _ds_init_eval_model(self, model):
        if not model:
            return model
        is_wrapper = self._is_wrapper(model)
        engine, *_ = deepspeed.initialize(
            model=self._extract_model(model),
            args={"local_rank": int(os.environ.get("LOCAL_RANK", "-1"))},
            config=self.get_ds_eval_config(offload=getattr(model, "_offload", False)),
            dist_init_required=True,
        )
        if is_wrapper:
            model.model = engine
        else:
            model = engine
        return model

    def get_ds_eval_config(self, offload=False):
        ds_config = get_eval_ds_config(
            offload=offload,
            stage=self.stage if self.stage == 3 else 0,
            param_dtype=self.param_dtype,
        )
        ds_config["train_micro_batch_size_per_gpu"] = self.micro_train_batch_size
        ds_config["train_batch_size"] = self.train_batch_size
        return ds_config

    def load_model(self, model: nn.Module, path: str, map_location="cpu", strict: bool = False, key_replace_fn=None):
        unwrapped_model = self._unwrap_model(model)
        state_dict = torch.load(path, map_location=map_location)
        if key_replace_fn:
            state_dict = key_replace_fn(state_dict)
        unwrapped_model.load_state_dict(state_dict, strict=strict)

    def save_model(self, model: nn.Module, tokenizer, output_dir, **kwargs) -> None:
        if self.is_rank_0():
            os.makedirs(output_dir, exist_ok=True)

        model_to_save = self._unwrap_model(model)
        extracted_model = self._extract_model(model)

        if self.args.zero_stage > 2:
            output_state_dict = extracted_model._consolidated_16bit_state_dict()
        else:
            from deepspeed.checkpoint.utils import clone_tensors_for_torch_save

            output_state_dict = clone_tensors_for_torch_save(model_to_save.state_dict())

        if self.is_rank_0():
            state_dict_keys = set(model_to_save.state_dict().keys())
            output_state_dict_keys = set(output_state_dict.keys())

            if getattr(model_to_save.config, "tie_word_embeddings", False) and "lm_head.weight" in state_dict_keys:
                state_dict_keys.remove("lm_head.weight")

            assert state_dict_keys.issubset(
                output_state_dict_keys
            ), f"mismatch keys {output_state_dict_keys.symmetric_difference(state_dict_keys)}"

            model_to_save.save_pretrained(output_dir, state_dict=output_state_dict, **kwargs)
            model_to_save.config.to_json_file(os.path.join(output_dir, "config.json"))
            tokenizer.save_pretrained(output_dir)

        del output_state_dict
        gc.collect()
        torch_dist_barrier_and_cuda_sync()

    def all_reduce(self, data, op="mean"):
        assert op in ("mean", "max", "sum")
        if isinstance(data, dict):
            return {k: self.all_reduce(v, op) for k, v in data.items()}

        is_tensor = True
        if not isinstance(data, torch.Tensor):
            data = torch.tensor([data], dtype=torch.float32)
            is_tensor = False
        is_cpu_tensor = data.device.type == "cpu"

        if is_cpu_tensor:
            data = data.to(torch.cuda.current_device())
        if op == "mean":
            data /= self.world_size
        dist.all_reduce(data, op=dist.ReduceOp.MAX if op == "max" else dist.ReduceOp.SUM)
        if is_cpu_tensor:
            data = data.cpu()
        return data.item() if not is_tensor else data

    def all_gather(self, data):
        if isinstance(data, dict):
            return {k: self.all_gather(v) for k, v in data.items()}

        if not isinstance(data, torch.Tensor):
            data = torch.tensor([data], dtype=torch.float32)
        is_cpu_tensor = data.device.type == "cpu"

        ret = [torch.zeros_like(data).to(torch.cuda.current_device()) for _ in range(self.world_size)]
        dist.all_gather(ret, data.to(torch.cuda.current_device()))
        return torch.cat(ret).cpu() if is_cpu_tensor else torch.cat(ret)

    def print(self, *msg):
        if self.is_rank_0():
            print(*msg)

    def is_rank_0(self) -> bool:
        if not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def get_rank(self) -> int:
        if not dist.is_initialized():
            return 0
        return dist.get_rank()

    def save_ckpt(self, model, save_dir, tag=None, max_num=3, max_mem=1000, client_state={}, save_latest=True):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        if self.is_rank_0():
            os.makedirs(save_dir, exist_ok=True)
            max_size = max_mem * 1024**3

            while True:
                subdirs = sorted(
                    [
                        (os.path.join(save_dir, d), os.path.getmtime(os.path.join(save_dir, d)))
                        for d in os.listdir(save_dir)
                        if os.path.isdir(os.path.join(save_dir, d))
                    ],
                    key=lambda x: x[1],
                )
                total_size = sum(
                    os.path.getsize(os.path.join(dirpath, f))
                    for subdir, _ in subdirs
                    for dirpath, _, filenames in os.walk(subdir)
                    for f in filenames
                )

                if len(subdirs) >= max_num or total_size > max_size:
                    oldest_dir = subdirs[0][0]
                    if os.path.exists(oldest_dir):
                        shutil.rmtree(oldest_dir)
                        self.print(f"Deleted oldest ckpt {oldest_dir}")
                else:
                    break

        torch_dist_barrier_and_cuda_sync()
        model.save_checkpoint(save_dir, tag=tag, client_state=client_state, save_latest=save_latest)
        gc.collect()

    def load_ckpt(
        self,
        model,
        load_dir,
        tag=None,
        load_module_strict=True,
        load_optimizer_states=True,
        load_lr_scheduler_states=True,
        load_module_only=False,
    ):
        assert isinstance(model, deepspeed.DeepSpeedEngine)
        load_path, states = model.load_checkpoint(
            load_dir,
            tag,
            load_module_strict=load_module_strict,
            load_optimizer_states=load_optimizer_states,
            load_lr_scheduler_states=load_lr_scheduler_states,
            load_module_only=load_module_only,
        )
        if load_path is None:
            raise Exception(f"[deepspeed] failed to resume from checkpoint {load_dir}")
        return load_path, states
