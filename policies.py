"""BC neural net policies."""

import abc

import numpy as np
import torch
import torch.nn as nn

TensorType = torch.Tensor


class BasePolicy(abc.ABC, nn.Module):
    """Base policy abstraction."""

    def __init__(self, action_range):
        super().__init__()

        self.action_range = action_range

    @abc.abstractmethod
    def forward(self, x: TensorType) -> TensorType:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        action_tensor = self.forward(obs_tensor)
        action_tensor = action_tensor.clamp(*self.action_range)
        action = action_tensor[0].cpu().detach().numpy()
        return action


class MLPPolicy(BasePolicy):
    """An MLP policy."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        if hidden_depth == 0:
            mods = [nn.Linear(input_dim, output_dim)]
        else:
            mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
            for _ in range(hidden_depth - 1):
                mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
            mods.append(nn.Linear(hidden_dim, output_dim))

        self.trunk = nn.Sequential(*mods)

        # Custom weight init.
        def _weight_init(m):
            """Custom weight init."""
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(_weight_init)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.trunk(x)
