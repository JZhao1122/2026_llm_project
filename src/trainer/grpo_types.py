from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch

from ..utils.utils import zero_pad_sequences


def _concat_optional_tensors(values: List[Optional[torch.Tensor]], pad_value: int = 0):
    tensors = [value for value in values if value is not None]
    if not tensors:
        return None
    stack = tensors[0].dim() == 1
    return zero_pad_sequences(tensors, side="right", value=pad_value, stack=stack)


def _concat_info(values: List[Any]):
    if not values:
        return values
    first = values[0]
    if isinstance(first, torch.Tensor):
        tensors = [value.reshape(-1) for value in values]
        return torch.cat(tensors, dim=0)
    if isinstance(first, list):
        merged = []
        for value in values:
            merged.extend(value)
        return merged
    return values


@dataclass
class GRPOExperience:
    sequences: torch.Tensor
    attention_mask: torch.Tensor
    action_mask: torch.Tensor
    old_action_log_probs: Optional[torch.Tensor] = None
    base_action_log_probs: Optional[torch.Tensor] = None
    advantages: Optional[torch.Tensor] = None
    returns: Optional[torch.Tensor] = None
    rewards: Optional[torch.Tensor] = None
    prompts: List[str] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    info: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return self.sequences.size(0)

    def to_device(self, device: torch.device):
        for name in (
            "sequences",
            "attention_mask",
            "action_mask",
            "old_action_log_probs",
            "base_action_log_probs",
            "advantages",
            "returns",
            "rewards",
        ):
            value = getattr(self, name)
            if isinstance(value, torch.Tensor):
                setattr(self, name, value.to(device))
        self.info = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in self.info.items()}
        return self

    def select(self, indices: torch.Tensor) -> "GRPOExperience":
        if indices.device.type != "cpu":
            indices = indices.cpu()
        idx_list = indices.tolist()

        def _select(value):
            if value is None:
                return None
            if isinstance(value, torch.Tensor):
                return value.index_select(0, indices)
            if isinstance(value, list):
                return [value[i] for i in idx_list]
            return value

        return GRPOExperience(
            sequences=self.sequences.index_select(0, indices),
            attention_mask=self.attention_mask.index_select(0, indices),
            action_mask=self.action_mask.index_select(0, indices),
            old_action_log_probs=_select(self.old_action_log_probs),
            base_action_log_probs=_select(self.base_action_log_probs),
            advantages=_select(self.advantages),
            returns=_select(self.returns),
            rewards=_select(self.rewards),
            prompts=_select(self.prompts),
            labels=_select(self.labels),
            info={key: _select(value) for key, value in self.info.items()},
        )

    @staticmethod
    def concat(experiences: List["GRPOExperience"], pad_token_id: int) -> "GRPOExperience":
        if not experiences:
            raise ValueError("Cannot concatenate an empty experience list")

        stack = experiences[0].sequences.dim() == 1
        info = {}
        keys = set()
        for experience in experiences:
            keys.update(experience.info.keys())
        for key in sorted(keys):
            info[key] = _concat_info([experience.info[key] for experience in experiences if key in experience.info])

        return GRPOExperience(
            sequences=zero_pad_sequences(
                [experience.sequences for experience in experiences],
                side="right",
                value=pad_token_id,
                stack=stack,
            ),
            attention_mask=zero_pad_sequences(
                [experience.attention_mask for experience in experiences],
                side="right",
                value=0,
                stack=stack,
            ),
            action_mask=zero_pad_sequences(
                [experience.action_mask for experience in experiences],
                side="right",
                value=0,
                stack=stack,
            ).bool(),
            old_action_log_probs=_concat_optional_tensors([experience.old_action_log_probs for experience in experiences], pad_value=0),
            base_action_log_probs=_concat_optional_tensors([experience.base_action_log_probs for experience in experiences], pad_value=0),
            advantages=_concat_optional_tensors([experience.advantages for experience in experiences], pad_value=0),
            returns=_concat_optional_tensors([experience.returns for experience in experiences], pad_value=0),
            rewards=(torch.cat([experience.rewards.reshape(-1) for experience in experiences], dim=0) if experiences[0].rewards is not None else None),
            prompts=sum([experience.prompts for experience in experiences], []),
            labels=sum([experience.labels for experience in experiences], []),
            info=info,
        )
