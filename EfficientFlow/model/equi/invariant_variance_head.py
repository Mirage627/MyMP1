"""Scalar uncertainty prediction from regular-representation features.

This module deliberately depends only on PyTorch so that its invariance and
gradient isolation can be checked without the optional robotics dependencies.
"""

import torch
from torch import nn


class InvariantVarianceHead(nn.Module):
    """Predict log mean-squared velocity residual for each trajectory.

    Input features have shape ``[batch, horizon, channels, group_order]``.
    A cyclic rotation permutes the last axis. Its first and second moments are
    invariant under that permutation. Pooling over the action horizon produces
    one scalar per trajectory; the head does not claim to estimate a full
    covariance or uncertainty of task success.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        for name, value in (("input_dim", input_dim), ("hidden_dim", hidden_dim)):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(2 * input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )
        # Start with variance one while leaving every existing velocity weight
        # unchanged when the head is attached to an existing model.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor, detach_features: bool = True):
        if not torch.is_tensor(features) or features.ndim != 4:
            raise ValueError("features must have shape [B, T, C, N]")
        if features.shape[2] != self.input_dim or any(size <= 0 for size in features.shape):
            raise ValueError("features have an empty axis or an incompatible channel dimension")
        if not features.is_floating_point():
            raise ValueError("features must be floating-point tensors")
        if detach_features:
            features = features.detach()
        moments = torch.cat((features.mean(dim=-1), features.square().mean(dim=-1)), dim=-1)
        return self.net(moments.mean(dim=1)).squeeze(-1)
