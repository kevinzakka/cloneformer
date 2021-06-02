"""Test a trained policy."""

import json
import os

import torch
import yaml
from absl import app, flags
from ml_collections import config_flags
from torchkit import checkpoint
from torchkit.utils.torch_utils import get_total_params

import evaluation
import utils

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_path", None, "Path to trained checkpoint.")
flags.DEFINE_integer("n_rollouts", 10, "Number of test rollouts.")

config_flags.DEFINE_config_file(
    "config",
    "config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

flags.mark_flag_as_required("experiment_path")


def load_experiment(exp_dir: str):
    """Initializes a training experiment."""
    with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
        cfg = yaml.load(fp, Loader=yaml.FullLoader)
    FLAGS.config.update(cfg)


def main(_):
    load_experiment(FLAGS.experiment_path)

    # Dynamically set observation and action space values.
    env = utils.load_xmagical_env(FLAGS.config)
    FLAGS.config.policy.input_dim = env.observation_space.shape[0]
    FLAGS.config.policy.output_dim = env.action_space.shape[0]
    FLAGS.config.policy.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]

    # Setup compute device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU {torch.cuda.get_device_name(device)}.")
    else:
        print("No GPU found. Falling back to CPU.")
        device = torch.device("cpu")

    policy = utils.get_policy(FLAGS.config).to(device)
    policy.eval()
    get_total_params(policy)

    # Load latest checkpoint.
    checkpoint_dir = os.path.join(FLAGS.experiment_path, "checkpoints")
    checkpoint_manager = checkpoint.CheckpointManager(
        checkpoint.Checkpoint(policy=policy),
        checkpoint_dir,
        device,
    )
    checkpoint_manager.restore_or_initialize()

    # Evaluate and dump result to disk.
    if FLAGS.config.policy.type == "transformer":
        protocol = evaluation.AEEvaluationProtocol(policy, FLAGS.n_rollouts, device)
    else:
        protocol = evaluation.EvaluationProtocol(policy, FLAGS.n_rollouts, device)
    result = protocol.do_eval(env)
    for key, val in result._asdict().items():
        print(f"{key}: {val:4f}")
    with open(os.path.join(FLAGS.experiment_path, "result.json"), "w") as fp:
        json.dump(result._asdict(), fp)


if __name__ == "__main__":
    app.run(main)
