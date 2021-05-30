"""BC neural net policies."""

import abc
import functools
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

TensorType = torch.Tensor


def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: int,
    dropout_prob: Optional[float] = 0.0,
    activation_fn=functools.partial(nn.ReLU, inplace=True),
    output_mod=None,
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
    if output_mod is not None:
        if not isinstance(output_mod, list):
            mods.append(output_mod)
        else:
            mods.extend(output_mod)
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
        mlp_hidden_dim: int,
        mlp_hidden_depth: int,
        mlp_dropout_prob: float,
        lstm_hidden_dim: int,
        output_dim: int,
        lstm_hidden_depth: int,
        lstm_dropout_prob: float,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mlp = build_mlp(
            input_dim,
            mlp_hidden_dim,
            lstm_hidden_dim,
            mlp_hidden_depth,
            dropout_prob=mlp_dropout_prob,
            output_mod=[nn.ReLU(inplace=True), nn.Dropout(mlp_dropout_prob)],
        )
        self.norm = nn.LayerNorm(lstm_hidden_dim)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_dim,
            hidden_size=lstm_hidden_dim,
            num_layers=lstm_hidden_depth,
            batch_first=True,
            dropout=lstm_dropout_prob if lstm_hidden_depth > 1 else 0,
        )
        self.head = nn.Linear(lstm_hidden_dim, output_dim)

    def forward(self, x_list: Sequence[TensorType]) -> Sequence[TensorType]:
        # Forward through the MLP and layer norm.
        out = [self.norm(self.mlp(x)) for x in x_list]

        # Pad and pack.
        seq_lens = [len(o) for o in out]
        out = pad_sequence(out, batch_first=True)
        out = pack_padded_sequence(out, lengths=seq_lens, batch_first=True)

        # Forward through LSTM.
        out, (ht, ct) = self.lstm(out)

        # Unpack sequence and remove padding.
        out, input_sizes = pad_packed_sequence(out, batch_first=True)
        out_trunk = []
        for out_batch, size in zip(out, input_sizes):
            out_trunk.append(out_batch[:size])

        # Forward through final projection head.
        out = [self.head(o) for o in out_trunk]

        return out

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
        obs_tensor = obs_tensor.to(next(self.parameters()).device)
        obs_tensor_list = [obs_tensor]
        action_tensor = self.forward(obs_tensor_list)[0]
        action_tensor = action_tensor.clamp(*self.action_range)
        action = action_tensor[0].cpu().detach().numpy()
        return action
