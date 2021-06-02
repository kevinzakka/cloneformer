"""BC neural net policies."""

import abc
import functools
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence,
                                pad_sequence)

TensorType = torch.Tensor


# Reference: https://github.com/denisyarats/drq
def build_mlp(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    hidden_depth: int,
    dropout_prob: Optional[float] = 0.0,
    activation_fn=functools.partial(nn.ReLU, inplace=True),
    output_mod=None,
):
    """Flexible MLP function creation."""
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

        out_mod = [nn.ReLU(inplace=True)]
        if mlp_dropout_prob > 0.0:
            out_mod.append(nn.Dropout(mlp_dropout_prob))
        self.mlp = build_mlp(
            input_dim,
            mlp_hidden_dim,
            lstm_hidden_dim,
            mlp_hidden_depth,
            dropout_prob=mlp_dropout_prob,
            output_mod=out_mod,
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

    def forward(self, x: TensorType) -> TensorType:
        out = self.norm(self.mlp(x))
        out, _ = self.lstm(out)
        out = self.head(out)
        return out

    def act(self, obs: np.ndarray) -> np.ndarray:
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).unsqueeze(0).float()
        obs_tensor = obs_tensor.to(next(self.parameters()).device)
        action_tensor = self.forward(obs_tensor)
        action_tensor = action_tensor.clamp(*self.action_range)
        action = action_tensor[0, 0].cpu().detach().numpy()
        return action


# Reference: https://github.com/karpathy/minGPT
class TransformerBlock(nn.Module):
    """Transformer with self-attention."""

    def __init__(
        self,
        emb_dim: int,
        num_heads: int,
        seq_len: int,
        dropout_prob: float = 0.1,
    ) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(emb_dim)
        self.ln2 = nn.LayerNorm(emb_dim)
        self.attn = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout_prob)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 4 * emb_dim),
            nn.GELU(),
            nn.Linear(4 * emb_dim, emb_dim),
            nn.Dropout(dropout_prob),
        )

    def forward(self, x: TensorType) -> TensorType:
        with torch.no_grad():
            k = x.shape[1]
            mask = torch.tril(torch.ones(k, k)).to(x.device)
            mask.detach()
        x_n = self.ln1(x)
        attn_out = self.attn(
            x_n.permute(1, 0, 2),
            x_n.permute(1, 0, 2),
            x_n.permute(1, 0, 2),
            attn_mask=mask,
        )[0]
        x = x + attn_out.permute(1, 0, 2)
        x = x + self.mlp(self.ln2(x))
        return x


# Reference: https://github.com/karpathy/minGPT
class AutoregressiveTransformerPolicy(BasePolicy):
    """A GPT-like Transformer policy."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        emb_dim: int,
        seq_len: int,
        num_blocks: int,
        num_heads: int,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.seq_len = seq_len

        # Embedding heads.
        self.s_emb = nn.Linear(state_dim, emb_dim)
        self.a_emb = nn.Linear(action_dim, emb_dim)

        # Timestep encoding.
        self.t_emb = nn.Parameter(torch.zeros(1, seq_len, emb_dim))

        # Transformer block.
        self.xformer = nn.Sequential(
            *[TransformerBlock(emb_dim, num_heads, seq_len) for _ in range(num_blocks)]
        )

        self.drop = nn.Dropout(0.1)

        # Layer normalization.
        self.s_ln = nn.LayerNorm(emb_dim)
        self.a_ln = nn.LayerNorm(emb_dim)

        # Output heads.
        self.a_head = nn.Linear(emb_dim, action_dim, bias=False)
        self.s_head = nn.Linear(emb_dim, state_dim, bias=False)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, learning_rate, betas, weight_decay):
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear,)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add("t_emb")
        decay.add("xformer.1.attn.in_proj_weight")
        decay.add("xformer.0.attn.in_proj_weight")

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert (
            len(inter_params) == 0
        ), "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert (
            len(param_dict.keys() - union_params) == 0
        ), "parameters %s were not separated into either decay/no_decay set!" % (
            str(param_dict.keys() - union_params),
        )

        # create the pytorch optimizer object
        optim_groups = [
            {
                "params": [param_dict[pn] for pn in sorted(list(decay))],
                "weight_decay": weight_decay,
            },
            {
                "params": [param_dict[pn] for pn in sorted(list(no_decay))],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def forward(self, x: TensorType, sample: bool = False) -> TensorType:
        if sample:
            with torch.no_grad():
                state_seq, action_seq = x
                if len(action_seq) == 0:
                    state = torch.cat(state_seq, dim=0).unsqueeze(0)
                    state_emb = self.s_ln(self.s_emb(state))
                    state_pos_emb = self.drop(
                        state_emb + self.t_emb[:, : state.shape[1], :]
                    )
                    out = self.xformer(state_pos_emb)
                    a_out = self.a_head(out)
                else:
                    state = torch.cat(state_seq, dim=0).unsqueeze(0)
                    action = torch.cat(action_seq, dim=0).unsqueeze(0)
                    state = state[:, -self.seq_len :]
                    action = action[:, -self.seq_len :]
                    state_emb = self.s_ln(self.s_emb(state))
                    action_emb = self.a_ln(self.a_emb(action))
                    state_pos_emb = self.drop(
                        state_emb + self.t_emb[:, : state.shape[1], :]
                    )
                    action_pos_emb = self.drop(
                        action_emb + self.t_emb[:, : action.shape[1], :]
                    )
                    state_action_pos_embs = []
                    for i in range(action_pos_emb.shape[1]):
                        state_action_pos_embs.append(state_pos_emb[:, i : i + 1, :])
                        state_action_pos_embs.append(action_pos_emb[:, i : i + 1, :])
                    state_action_pos_embs.append(state_pos_emb[:, i + 1 : i + 2, :])
                    state_action_pos_embs = torch.cat(state_action_pos_embs, dim=1)
                    out = self.xformer(state_action_pos_embs)
                    s_out = []
                    a_out = []
                    for i in range(out.shape[1]):
                        if not i % 2:
                            a_out.append(out[:, i : i + 1, :])
                        else:
                            s_out.append(out[:, i : i + 1, :])
                    s_out = torch.cat(s_out, dim=1)
                    a_out = torch.cat(a_out, dim=1)
                    s_out = self.s_head(s_out)
                    a_out = self.a_head(a_out)
                    a_out = a_out[:, -1:, :]
                return a_out
        else:
            state, action = x
            state_i = state
            state_o = state[:, 1:, :]
            action_i = action[:, :-1, :]
            action_o = action
            state_emb = self.s_ln(self.s_emb(state_i))
            action_emb = self.a_ln(self.a_emb(action_i))
            state_pos_emb = self.drop(state_emb + self.t_emb[:, : state_i.shape[1], :])
            action_pos_emb = self.drop(
                action_emb + self.t_emb[:, : action_i.shape[1], :]
            )
            state_action_pos_embs = []
            for i in range(action_pos_emb.shape[1]):
                state_action_pos_embs.append(state_pos_emb[:, i : i + 1, :])
                state_action_pos_embs.append(action_pos_emb[:, i : i + 1, :])
            state_action_pos_embs.append(state_pos_emb[:, i + 1 : i + 2, :])
            state_action_pos_embs = torch.cat(state_action_pos_embs, dim=1)
            out = self.xformer(state_action_pos_embs)
            s_out = []
            a_out = []
            for i in range(out.shape[1]):
                if not i % 2:
                    a_out.append(out[:, i : i + 1, :])
                else:
                    s_out.append(out[:, i : i + 1, :])
            s_out = torch.cat(s_out, dim=1)
            a_out = torch.cat(a_out, dim=1)
            s_out = self.s_head(s_out)
            a_out = self.a_head(a_out)
            assert a_out.shape == action_o.shape
            assert s_out.shape == state_o.shape
            return (s_out, a_out), (state_o, action_o)
