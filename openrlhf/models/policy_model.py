from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig

from .utils import compute_entropy, log_probs_from_logits


class PolicyModel(nn.Module):
    """Causal LM wrapper used by GRPO for rollout and policy updates."""

    def __init__(
        self,
        model_name_or_path,
        attn_implementation="flash_attention_2",
        param_dtype="bf16",
        ds_config=None,
        device_map=None,
        temperature=1.0,
    ) -> None:
        super().__init__()
        self.temperature = temperature
        self._ds_config_helper = None

        if isinstance(model_name_or_path, str):
            if ds_config is not None and ds_config["zero_optimization"]["stage"] == 3:
                self._ds_config_helper = HfDeepSpeedConfig(ds_config)

            from openrlhf.utils.utils import convert_to_torch_dtype

            torch_dtype = convert_to_torch_dtype(param_dtype)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                attn_implementation=attn_implementation,
                torch_dtype=torch_dtype,
                device_map=device_map,
            )
            self.model.config.use_cache = False
        else:
            self.model = model_name_or_path

    def forward(
        self,
        sequences: torch.LongTensor,
        action_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        return_output=False,
        return_entropy=False,
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(sequences, dtype=torch.long)

        rolled_sequences = torch.roll(sequences, shifts=-1, dims=1)
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)

        output = self.model(sequences, attention_mask=attention_mask, position_ids=position_ids)
        output["logits"] = output["logits"].to(torch.float32)

        if return_entropy:
            assert return_output
            setattr(output, "entropy", compute_entropy(output["logits"])[:, :-1])

        log_probs = log_probs_from_logits(output["logits"], rolled_sequences, temperature=self.temperature)[:, :-1]
        if action_mask is None:
            return (log_probs, output) if return_output else log_probs

        action_log_probs = log_probs[:, -action_mask.shape[1] :] * action_mask.float()
        return (action_log_probs, output) if return_output else action_log_probs

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()
