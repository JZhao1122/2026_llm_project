from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


def compute_approx_kl(
    log_probs: torch.Tensor,
    log_probs_base: torch.Tensor,
    kl_estimator: str = "k1",  # k1/k2/k3
) -> torch.Tensor:
    """
    TODO: Compute unreduced approximate KL values between the current policy
    and the base policy. Please refer to Schulman blog
    (http://joschu.net/blog/kl-approx.html) for the three estimators: k1, k2, and k3.

    Args:
        log_probs: Log probabilities from the current policy. [batch_size, len]
        log_probs_base: Log probabilities from the base/reference policy. [batch_size, len]
        kl_estimator: Which approximation to use. One of {"k1", "k2", "k3"}.

    Returns:
        torch.Tensor: Approximate per-token KL values with the same shape as
            `log_probs`. This function does not reduce across batch or sequence.
    """
    # ====== YOUR CODE HERE ======
    raise NotImplementedError()
    # ====== END YOUR CODE ======
