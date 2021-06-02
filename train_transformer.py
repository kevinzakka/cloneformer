"""Autoregressive behavior cloning with Transformers."""

import os

import torch
import torch.nn.functional as F
import yaml
from absl import app, flags
from ipdb import set_trace
from ml_collections import ConfigDict, config_flags
from torchkit import Logger, checkpoint
from torchkit.utils.torch_utils import get_total_params

import utils
import video

FLAGS = flags.FLAGS

flags.DEFINE_string("experiment_name", None, "Experiment name.")
flags.DEFINE_boolean("resume", False, "Whether to resume training.")

config_flags.DEFINE_config_file(
    "config",
    "config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

flags.mark_flag_as_required("experiment_name")


def setup_experiment(exp_dir: str):
    """Initializes a training experiment."""
    if os.path.exists(exp_dir):
        if not FLAGS.resume:
            raise ValueError(
                "Experiment already exists. Run with --resume to continue."
            )
        with open(os.path.join(exp_dir, "config.yaml"), "r") as fp:
            cfg = yaml.load(fp, Loader=yaml.FullLoader)
        FLAGS.config.update(cfg)
    else:
        os.makedirs(exp_dir)
        with open(os.path.join(exp_dir, "config.yaml"), "w") as fp:
            yaml.dump(ConfigDict.to_dict(FLAGS.config), fp)


@torch.no_grad()
def eval_policy(policy, valid_loader, device) -> float:
    policy.eval()
    valid_loss = 0.0
    for state, action in valid_loader:
        state, action = state.to(device), action.to(device)
        (s_pred, a_pred), (s_gt, a_gt) = policy([state, action])
        action_loss = (
            F.mse_loss(a_pred, a_gt, reduction="none").sum(dim=-1).sum(dim=-1).mean()
        )
        state_loss = (
            F.mse_loss(s_pred, s_gt, reduction="none").sum(dim=-1).sum(dim=-1).mean()
        )
        valid_loss += action_loss + state_loss
    valid_loss /= len(valid_loader.dataset)
    print(f"Validation loss: {valid_loss:.6f}")
    return valid_loss


@torch.no_grad()
def rollout_policy(policy, video_recorder, global_step, env, device):
    policy.eval()
    average_episode_success = 0
    for episode in range(FLAGS.config.num_eval_episodes):
        env.seed()
        obs = env.reset()
        video_recorder.reset(enabled=(episode == 0))
        state_seq = [torch.from_numpy(obs).unsqueeze(0).float().to(device)]
        action_seq = []
        while True:
            action_tensor = policy([state_seq, action_seq], sample=True)
            action_tensor = action_tensor.clamp(*policy.action_range)
            action_tensor = action_tensor[:, -1, :]
            action = action_tensor[0].cpu().detach().numpy()
            obs, _, done, info = env.step(action)
            state_seq.append(torch.from_numpy(obs).unsqueeze(0).float().to(device))
            action_seq.append(action_tensor)
            video_recorder.record(env)
            if done:
                average_episode_success += info["eval_score"]
                break
        video_recorder.save(f"{global_step}.mp4")
    average_episode_success /= FLAGS.config.num_eval_episodes
    print(f"Avg episode success: {average_episode_success:.4f}")
    return average_episode_success


def main(_):
    exp_dir = os.path.join(FLAGS.config.root_dir, FLAGS.experiment_name)
    setup_experiment(exp_dir)

    env = utils.load_xmagical_env(FLAGS.config)

    # Dynamically set observation and action space values.
    FLAGS.config.policy.input_dim = env.observation_space.shape[0]
    FLAGS.config.policy.output_dim = env.action_space.shape[0]
    FLAGS.config.policy.action_range = [
        float(env.action_space.low.min()),
        float(env.action_space.high.max()),
    ]

    # Set RNG seeds.
    if FLAGS.config.seed is not None:
        print(f"Experiment seed: {FLAGS.config.seed}.")
        utils.seed_rng(FLAGS.config)
    else:
        print("No RNG seed has been set for this experiment.")

    # Setup compute device.
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU {torch.cuda.get_device_name(device)}.")
    else:
        print("No GPU found. Falling back to CPU.")
        device = torch.device("cpu")

    logger = Logger(exp_dir, FLAGS.resume)
    video_recorder = video.VideoRecorder(exp_dir if FLAGS.config.save_video else None)

    data_loaders = utils.get_bc_dataloaders(FLAGS.config)
    num_train_pairs = len(data_loaders["train"].dataset)
    num_valid_pairs = len(data_loaders["valid"].dataset)
    print(f"Training on {num_train_pairs} state, action pairs.")
    print(f"Validating on {num_valid_pairs} state, action pairs.")

    policy = utils.get_policy(FLAGS.config).to(device)
    print(policy)
    get_total_params(policy)
    optimizer = policy.configure_optimizers(
        FLAGS.config.learning_rate, (0.9, 0.95), FLAGS.config.l2_reg
    )

    # Create checkpoint manager.
    checkpoint_dir = os.path.join(exp_dir, "checkpoints")
    checkpoint_manager = checkpoint.CheckpointManager(
        checkpoint.Checkpoint(policy=policy, optimizer=optimizer),
        checkpoint_dir,
        device,
    )

    complete = False
    global_step = checkpoint_manager.restore_or_initialize()
    epoch = 0
    try:
        while not complete:
            for state, action in data_loaders["train"]:
                state, action = state.to(device), action.to(device)

                policy.train()
                optimizer.zero_grad()
                (s_pred, a_pred), (s_gt, a_gt) = policy([state, action])
                action_loss = (
                    F.mse_loss(a_pred, a_gt, reduction="none")
                    .sum(dim=-1)
                    .sum(dim=-1)
                    .mean()
                )
                state_loss = (
                    F.mse_loss(s_pred, s_gt, reduction="none")
                    .sum(dim=-1)
                    .sum(dim=-1)
                    .mean()
                )
                loss = action_loss + state_loss
                loss.backward()
                optimizer.step()

                if not global_step % FLAGS.config.logging_frequency:
                    logger.log_scalar(loss, global_step, "train/loss")
                    print(
                        "Iter[{}/{}] (Epoch {}), Loss: {:.3f}".format(
                            global_step,
                            FLAGS.config.train_max_iters,
                            epoch,
                            loss.item(),
                        )
                    )

                if not global_step % FLAGS.config.eval_frequency:
                    valid_loss = eval_policy(policy, data_loaders["valid"], device)
                    logger.log_scalar(valid_loss, global_step, "valid/loss")
                    eval_success = rollout_policy(
                        policy,
                        video_recorder,
                        global_step,
                        env,
                        device,
                    )
                    logger.log_scalar(eval_success, global_step, "valid/success")

                # Save model checkpoint.
                if not global_step % FLAGS.config.checkpoint_frequency:
                    checkpoint_manager.save(global_step)

                # Exit if complete.
                global_step += 1
                if global_step > FLAGS.config.train_max_iters:
                    complete = True
                    break
            epoch += 1

    except KeyboardInterrupt:
        print("Caught keyboard interrupt. Saving model before quitting.")

    finally:
        checkpoint_manager.save(global_step)
        logger.close()


if __name__ == "__main__":
    app.run(main)
