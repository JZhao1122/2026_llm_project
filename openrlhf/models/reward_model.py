from typing import Optional

import deepspeed
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from transformers.integrations.deepspeed import HfDeepSpeedConfig


class RewardModel(nn.Module):
    """Sequence-level reward model used by local GRPO."""

    def __init__(
        self,
        model_name_or_path: str,
        *,
        param_dtype="bf16",
        normalize_reward=False,
        attn_implementation="flash_attention_2",
        ds_config: dict = None,
        init_value_head=False,
        value_head_prefix="score",
        device_map=None,
    ) -> None:
        super().__init__()
        self._ds_config_helper = None

        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
        config.normalize_reward = normalize_reward
        config._attn_implementation = attn_implementation
        self.value_head_prefix = getattr(config, "value_head_prefix", value_head_prefix)

        base_class = AutoModel._model_mapping[type(config)]
        base_pretrained_class = base_class.__base__
        model_class = self._get_reward_model(base_pretrained_class, base_class, self.value_head_prefix)

        if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
            self._ds_config_helper = HfDeepSpeedConfig(ds_config)

        from openrlhf.utils.utils import convert_to_torch_dtype

        torch_dtype = convert_to_torch_dtype(param_dtype)
        self.model = model_class.from_pretrained(
            model_name_or_path,
            config=config,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.config.use_cache = False

        if init_value_head:
            value_head = getattr(self.model, self.value_head_prefix)
            if self._ds_config_helper is not None:
                with deepspeed.zero.GatheredParameters([value_head.weight], modifier_rank=0):
                    if torch.distributed.get_rank() == 0:
                        value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))
            else:
                value_head.weight.data.normal_(mean=0.0, std=1 / (config.hidden_size + 1))

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask)

    @staticmethod
    def _get_reward_model(base_pretrained_model, base_llm_model, value_head_prefix="score"):
        class RewardBackbone(base_pretrained_model):
            supports_gradient_checkpointing = True

            def __init__(self, config: AutoConfig):
                super().__init__(config)
                setattr(self, self.base_model_prefix, base_llm_model(config))
                setattr(self, value_head_prefix, nn.Linear(config.hidden_size, 1, bias=False))

                self.value_head_prefix = value_head_prefix
                self.normalize_reward = config.normalize_reward
                self.register_buffer("mean", torch.zeros(1), persistent=False)
                self.register_buffer("std", torch.ones(1), persistent=False)
                if hasattr(config, "mean"):
                    self.mean[0] = config.mean
                    self.std[0] = config.std
                self.post_init()

            def forward(
                self,
                input_ids: torch.LongTensor = None,
                attention_mask: Optional[torch.Tensor] = None,
            ) -> torch.Tensor:
                eos_indices = attention_mask.size(1) - 1 - attention_mask.long().fliplr().argmax(dim=1, keepdim=True)
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)

                outputs = getattr(self, self.base_model_prefix)(
                    input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                )
                last_hidden_states = outputs["last_hidden_state"]
                values = getattr(self, self.value_head_prefix)(last_hidden_states).squeeze(-1)
                reward = values.gather(dim=1, index=eos_indices).squeeze(1)
                if not self.training and self.normalize_reward:
                    reward = (reward - self.mean) / self.std
                return reward

        return RewardBackbone
