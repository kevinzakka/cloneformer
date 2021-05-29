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
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.batch_size,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
            shuffle=True,
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
    env_name = f"SweepToTop-{config.embodiment.capitalize()}-State-Allo-Demo-v0"
    env = gym.make(env_name)
    env = wrappers.wrapper_from_config(config, env)
    return env
