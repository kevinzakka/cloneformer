"""Default config variables."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()

    # ============================================== #
    # General experiment params.
    # ============================================== #
    # The root directory where experiments will be saved.
    config.ROOT_DIR = "/home/kevin/Desktop/bcformer/results/"

    # RNG seed. Set this to `None` to disable seeding.
    config.SEED = 1

    # CUDNN-related parameters that affect reproducibility.
    config.CUDNN_DETERMINISTIC = False
    config.CUDNN_BENCHMARK = True

    config.EMBODIMENT = "gripper"

    # ============================================== #
    # Dataset params.
    # ============================================== #
    config.DATA = ml_collections.ConfigDict()

    # Absolute path to the dataset root.
    config.DATA.ROOT = "/home/kevin/datasets/xirl_corl/"

    # ============================================== #
    # End of config file
    # ============================================== #

    return config
