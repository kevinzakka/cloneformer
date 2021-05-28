from absl import app
from absl import flags

from ml_collections.config_flags import config_flags

import utils

from ipdb import set_trace

FLAGS = flags.FLAGS

# flags.DEFINE_string("experiment_name", None, "Experiment name.")

config_flags.DEFINE_config_file(
    "config",
    "config.py",
    "File path to the training hyperparameter configuration.",
    lock_config=True,
)

# flags.mark_flag_as_required("experiment_name")


def main(_):
    dloaders = utils.get_bc_dataloaders(FLAGS.config)

    set_trace()


if __name__ == "__main__":
  app.run(main)
