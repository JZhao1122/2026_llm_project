from typing import Optional

import torch
import torch.nn as nn


class SFTLoss(nn.Module):
    """Masked negative log-prob loss for supervised fine-tuning."""

    def __init__(self):
        super().__init__()

    def forward(self, per_token_logps: torch.Tensor, loss_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        '''
        TODO: Return the negative log-prob loss (averaged over tokens).
        return a scalar tensor
        '''
        # ====== YOUR CODE HERE ======
        raise NotImplementedError()
        # ====== END YOUR CODE ======
