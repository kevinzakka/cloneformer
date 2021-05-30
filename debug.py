"""Debugging."""

import torch
from absl import app, flags
from ipdb import set_trace
from ml_collections import config_flags
from torchkit.utils.torch_utils import get_total_params

import utils

FLAGS = flags.FLAGS

config_flags.DEFINE_config_file(
    "config",
    "config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)


def main(_):
    env = utils.load_xmagical_env(FLAGS.config)

    # Dynamically set observation and action space values.
    FLAGS.config.policy.input_dim = env.observation_space.shape[0]
    FLAGS.config.policy.output_dim = env.action_space.shape[0]
    FLAGS.config.policy.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]

    data_loaders = utils.get_bc_dataloaders(FLAGS.config)
    policy = utils.get_policy(FLAGS.config)
    print(policy)
    get_total_params(policy)

    for idx, (state, action) in enumerate(data_loaders["train"]):
        with torch.no_grad():
            out = policy(state)
        break


if __name__ == "__main__":
    app.run(main)
