import torch.nn as nn
from transformers import AutoModelForCausalLM
from transformers.integrations.deepspeed import HfDeepSpeedConfig


class SFTModel(nn.Module):
    """Minimal causal LM wrapper used only for SFT."""

    def __init__(
        self,
        model_name_or_path,
        attn_implementation="flash_attention_2",
        param_dtype="bf16",
        ds_config=None,
        device_map=None,
        **kwargs
    ) -> None:
        super().__init__()
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

    def forward(self, input_ids, attention_mask, **kwargs):
        '''
        TODO: Forward pass that computes per-token log probabilities.
        Return: log_probs, [batch_size, seq_len - 1]
        '''
        # ====== YOUR CODE HERE ======
        raise NotImplementedError()
        # ====== END YOUR CODE ======

    def gradient_checkpointing_enable(self):
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.model.gradient_checkpointing_disable()
