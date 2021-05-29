"""Default config variables."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ================================================= #
    # Placeholders.
    # ================================================= #
    obs_dim = ml_collections.FieldReference(None, field_type=int)
    action_dim = ml_collections.FieldReference(None, field_type=int)
    action_range = ml_collections.FieldReference(None, field_type=tuple)

    # ============================================== #
    # Env params.
    # ============================================== #
    config.action_repeat = 1
    config.frame_stack = 3

    # ============================================== #
    # General experiment params.
    # ============================================== #
    config.embodiment = "gripper"

    # The root directory where experiments will be saved.
    config.root_dir = "/home/kevin/Desktop/bcformer/results/"

    # RNG seed. Set this to `None` to disable seeding.
    config.seed = 1

    # CUDNN-related parameters that affect reproducibility.
    config.cudnn_deterministic = False
    config.cudnn_benchmark = True

    # Whether to save eval rollouts to disk.
    config.save_video = True

    # ============================================== #
    # Dataset params.
    # ============================================== #
    config.data = ml_collections.ConfigDict()

    # Absolute path to the dataset root.
    config.data.root = "/home/kevin/datasets/xirl_corl/"

    # ================================================= #
    # Training parameters.
    # ================================================= #
    config.num_eval_episodes = 20
    config.train_max_iters = 100_000
    config.eval_frequency = 10_000
    config.logging_frequency = 1_000
    config.checkpoint_frequency = 25_000
    config.batch_size = 128
    config.learning_rate = 1e-3
    config.l2_reg = 1e-5

    # ============================================== #
    # Policy params.
    # ============================================== #
    config.policy = ml_collections.ConfigDict()

    config.policy.type = "mlp"
    config.policy.input_dim = obs_dim
    config.policy.output_dim = action_dim
    config.policy.action_range = action_range

    # MLP policy params.
    config.policy.mlp = ml_collections.ConfigDict()
    config.policy.mlp.hidden_dim = 128
    config.policy.mlp.hidden_depth = 3
    config.policy.mlp.dropout_prob = 0.1

    # LSTM policy params.
    config.policy.lstm = ml_collections.ConfigDict()
    config.policy.lstm.hidden_dim = 128
    config.policy.lstm.hidden_depth = 3

    # ============================================== #
    # End of config file
    # ============================================== #

    return config
