"""Common utilities."""

import os.path as osp
import random

import gym
import numpy as np
import torch

import data
import policies
import wrappers


def seed_rng(config) -> None:
    """Seed the RNGs across all modules."""
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = config.cudnn_deterministic
    torch.backends.cudnn.benchmark = config.cudnn_benchmark


def get_bc_dataloaders(config):
    """Construct a train/valid pair of pretraining dataloaders."""

    def _loader(split):
        dirname = osp.join(config.data.root, split, config.embodiment)
        dataset = data.BCDataset(dirname, from_state=True)
        if config.policy.type == "mlp":
            dataset = data.BCDataset(dirname, from_state=True)
        elif config.policy.type in ["lstm", "transformer"]:
            dataset = data.SequentialBCDataset(
                dirname, from_state=True, seq_len=config.seq_len,
                autoregressive=config.policy.type=="transformer",
            )
        else:
            raise ValueError(f"No dataset for {config.policy.type} policies.")
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
            shuffle=True if split == "train" else False,
            collate_fn=None,
        )

    return {
        "train": _loader("train"),
        "valid": _loader("valid"),
    }


def get_policy(config):
    if config.policy.type == "mlp":
        policy = policies.MLPPolicy(
            input_dim=config.policy.input_dim,
            hidden_dim=config.policy.mlp.hidden_dim,
            output_dim=config.policy.output_dim,
            hidden_depth=config.policy.mlp.hidden_depth,
            dropout_prob=config.policy.mlp.dropout_prob,
            action_range=config.policy.action_range,
        )
    elif config.policy.type == "lstm":
        policy = policies.LSTMPolicy(
            input_dim=config.policy.input_dim,
            mlp_hidden_dim=config.policy.lstm.mlp_hidden_dim,
            mlp_hidden_depth=config.policy.lstm.mlp_hidden_depth,
            mlp_dropout_prob=config.policy.lstm.mlp_dropout_prob,
            lstm_hidden_dim=config.policy.lstm.lstm_hidden_dim,
            output_dim=config.policy.output_dim,
            lstm_hidden_depth=config.policy.lstm.lstm_hidden_depth,
            lstm_dropout_prob=config.policy.lstm.lstm_dropout_prob,
            action_range=config.policy.action_range,
        )
    elif config.policy.type == "transformer":
        policy = policies.AutoregressiveTransformerPolicy(
            state_dim=config.policy.input_dim,
            action_dim=config.policy.output_dim,
            emb_dim=config.policy.xformer.emb_dim,
            num_blocks=config.policy.xformer.num_blocks,
            num_heads=config.policy.xformer.num_heads,
            seq_len=config.seq_len,
            action_range=config.policy.action_range,
        )
    else:
        raise ValueError(f"{config.policy.type} is not a supported policy.")
    return policy


def get_optimizer(config, model):
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.l2_reg,
    )


def load_xmagical_env(config) -> gym.Env:
    import xmagical

    xmagical.register_envs()
    env_name = f"SweepToTop-{config.embodiment.capitalize()}-State-Allo-TestLayout-v0"
    env = gym.make(env_name)
    env = wrappers.wrapper_from_config(config, env)
    return env
