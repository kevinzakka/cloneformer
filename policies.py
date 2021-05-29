"""BC neural net policies."""

import abc

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import distributions as pyd

TensorType = torch.Tensor


class BasePolicy(abc.ABC, nn.Module):
    def __init__(self, *, action_range):
        self.action_range = action_range

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.trunk(x)

    def act(self, obs: np.ndarray) -> np.ndarray:
        pass
        # action_tensor = action_tensor.clamp(*action_range)


class MLPPolicy(BasePolicy):
    """An MLP policy."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int,
    ) -> None:
        super().__init__()

        if hidden_depth == 0:
            mods = [nn.Linear(input_dim, output_dim)]
        else:
            mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
            for _ in range(hidden_depth - 1):
                mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
            mods.append(nn.Linear(hidden_dim, output_dim))

        self.trunk = nn.Sequential(*mods)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.trunk(x)
