from .loss import SFTLoss
from .sft_model import SFTModel

__all__ = ["PolicyLoss", "PolicyModel", "RewardModel", "SFTLoss", "SFTModel"]


def __getattr__(name):
    if name == "PolicyLoss":
        from openrlhf.models.loss import PolicyLoss

        return PolicyLoss
    if name == "PolicyModel":
        from openrlhf.models.policy_model import PolicyModel

        return PolicyModel
    if name == "RewardModel":
        from openrlhf.models.reward_model import RewardModel

        return RewardModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
