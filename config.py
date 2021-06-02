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
    config.num_eval_episodes = 100
    config.train_max_iters = 1_000_000
    config.eval_frequency = 10_000
    config.logging_frequency = 100
    config.checkpoint_frequency = 25_000
    config.batch_size = 128
    config.learning_rate = 1e-3
    config.l2_reg = 1e-5
    config.clip_grad_norm = 1.0

    # Used for transformer and LSTM training.
    config.seq_len = 5

    # ============================================== #
    # Policy params.
    # ============================================== #
    config.policy = ml_collections.ConfigDict()

    config.policy.type = "transformer"
    config.policy.input_dim = obs_dim
    config.policy.output_dim = action_dim
    config.policy.action_range = action_range

    # MLP policy params.
    config.policy.mlp = ml_collections.ConfigDict()
    config.policy.mlp.hidden_dim = 128
    config.policy.mlp.hidden_depth = 2
    config.policy.mlp.dropout_prob = 0.1

    # LSTM policy params.
    config.policy.lstm = ml_collections.ConfigDict()
    config.policy.lstm.mlp_hidden_dim = 128
    config.policy.lstm.mlp_hidden_depth = 2
    config.policy.lstm.mlp_dropout_prob = 0.1
    config.policy.lstm.lstm_hidden_dim = 64
    config.policy.lstm.lstm_hidden_depth = 1
    config.policy.lstm.lstm_dropout_prob = 0.2

    # Autoregressive Transformer policy params.
    config.policy.xformer = ml_collections.ConfigDict()
    config.policy.xformer.emb_dim = 64
    config.policy.xformer.num_blocks = 2
    config.policy.xformer.num_heads = 4

    # ============================================== #
    # End of config file
    # ============================================== #

    return config
