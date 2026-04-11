from typing import Optional

import torch
import torch.nn as nn

from .utils import masked_mean


class SFTLoss(nn.Module):
    def __init__(self, token_level_loss: bool = True):
        super().__init__()
        self.token_level_loss = token_level_loss

    def forward(self, per_token_logps: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
        if self.token_level_loss:
            return masked_mean(-per_token_logps, loss_mask, dim=None)
        return masked_mean(-per_token_logps, loss_mask, dim=-1).mean()


class PolicyLoss(nn.Module):
    def __init__(
        self,
        clip_eps_low: float = 0.2,
        clip_eps_high: float = 0.2,
        dual_clip: float = None,
        token_level_loss: bool = True,
    ) -> None:
        super().__init__()
        self.clip_eps_low = clip_eps_low
        self.clip_eps_high = clip_eps_high
        self.token_level_loss = token_level_loss
        self.dual_clip = dual_clip

        if dual_clip is not None:
            assert dual_clip > 1.0, f"dual_clip must be > 1.0, got {dual_clip}"

    def forward(
        self,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None,
    ):
        log_ratio = log_probs - old_log_probs
        ratio = log_ratio.exp()

        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps_low, 1 + self.clip_eps_high) * advantages

        if self.dual_clip is None:
            loss = -torch.min(surr1, surr2)
        else:
            clip1 = torch.min(surr1, surr2)
            clip2 = torch.max(clip1, self.dual_clip * advantages)
            loss = -torch.where(advantages < 0, clip2, clip1)

        if self.token_level_loss:
            loss = masked_mean(loss, action_mask, dim=None)
        else:
            loss = masked_mean(loss, action_mask, dim=-1).mean()

        clip_ratio = masked_mean(torch.lt(surr2, surr1).float(), action_mask, dim=None)
        approx_kl = masked_mean(-log_ratio.detach(), action_mask, dim=None)
        return loss, clip_ratio, approx_kl
