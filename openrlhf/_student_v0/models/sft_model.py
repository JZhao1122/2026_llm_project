import torch.nn as nn


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
        '''
        TODO: Init the model
        '''
        # ====== YOUR CODE HERE ======
        raise NotImplementedError()
        # ====== END YOUR CODE ======

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
