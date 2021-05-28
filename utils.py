"""Common utilities."""

import os.path as osp
import torch

from data import BCDataset


def get_bc_dataloaders(config):
    """Construct a train/valid pair of pretraining dataloaders."""

    def _loader(split):
        dirname = osp.join(config.DATA.ROOT, split, config.EMBODIMENT)
        dataset = BCDataset(dirname, from_state=True)
        return torch.utils.data.DataLoader(
            dataset,
            num_workers=4 if torch.cuda.is_available() else 0,
            pin_memory=torch.cuda.is_available(),
        )

    return {
        "train": _loader("train"),
        "valid": _loader("valid"),
    }
