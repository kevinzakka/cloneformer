"""BC neural net policies."""

import abc
import functools
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence

TensorType = torch.Tensor


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: int,
    dropout_prob: Optional[float] = 0.0,
    activation_fn=functools.partial(nn.ReLU, inplace=True),
):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
        if dropout_prob > 0.0:
            mods += [nn.Dropout(dropout_prob)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), activation_fn()]
        if dropout_prob > 0.0:
            mods += [nn.Dropout(dropout_prob)]
        for _ in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), activation_fn()]
            if dropout_prob > 0.0:
                mods += [nn.Dropout(dropout_prob)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


class BasePolicy(abc.ABC, nn.Module):
    """Base policy abstraction."""

    def __init__(self, action_range: Tuple[float, ...]) -> None:
        super().__init__()

        self.action_range = action_range

    @abc.abstractmethod
    def forward(self, x: TensorType) -> TensorType:
        pass

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        obs_tensor = obs_tensor.to(next(self.parameters()).device)
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
        dropout_prob: float,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.trunk = build_mlp(
            input_dim,
            hidden_dim,
            output_dim,
            hidden_depth,
            dropout_prob=dropout_prob,
        )

        # Custom weight init.
        def _weight_init(m):
            """Custom weight init."""
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data, nonlinearity="relu")
                if hasattr(m.bias, "data"):
                    m.bias.data.fill_(0.0)

        self.apply(_weight_init)

    def forward(self, x: TensorType) -> TensorType:
        return self.trunk(x)


class LSTMPolicy(BasePolicy):
    """An LSTM-policy."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        hidden_depth: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.trunk = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=hidden_depth,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: TensorType) -> TensorType:
        seq_lens = [len(b) for b in x]
        out = pad_sequence(x, batch_first=True)
        out = pack_padded_sequence(out, lengths=seq_lens, batch_first=True)
        out, ht = self.trunk(out)
        out = self.head(ht[-1])
        return out
