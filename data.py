"""Utilities for manipulating and loading expert demonstrations.

Large parts adapted from: https://github.com/HumanCompatibleAI/imitation
"""

import dataclasses
import glob
import json
import os
import os.path as osp
import random
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


@dataclasses.dataclass(frozen=True)
class Trajectory:
    """An episode rollout from an expert policy."""

    obs: np.ndarray  # (N + 1,)
    acts: np.ndarray  # (N,)
    infos: Optional[np.ndarray]  # (N,)

    def __post_init__(self):
        assert len(self.acts) > 0
        assert len(self.obs) == len(self.acts) + 1

    def __len__(self):
        return len(self.acts)

    @classmethod
    def load_from_folder(
        self,
        foldername: str,
        from_state: Optional[bool] = False,
        ext: Optional[str] = "*.png",
        resize_hw: Optional[Tuple[int, int]] = None,
    ) -> "Trajectory":
        """Construct a trajectory from a folder."""
        if from_state:
            # Load states.
            with open(osp.join(foldername, "states.json"), "r") as fp:
                obs = np.array(json.load(fp), dtype=np.float32)
        else:
            # Load observations.
            obs = glob.glob(os.path.join(foldername, ext))
            obs = sorted(obs, key=lambda x: int(osp.basename(x).split(".")[0]))
            obs = [Image.open(o) for o in obs]
            if resize_hw is not None:
                obs = [
                    o.resize((resize_hw[1], resize_hw[0]), Image.BILINEAR) for o in obs
                ]
            obs = np.stack(obs)

        # Load actions.
        with open(osp.join(foldername, "actions.json"), "r") as fp:
            acts = np.array(json.load(fp), dtype=np.float32)

        return Trajectory(obs=obs, acts=acts, infos=None)


class BCDataset(torch.utils.data.Dataset):
    """A dataset of trajectories for behavior cloning."""

    def __init__(self, dirname: str, from_state: bool) -> None:
        # Get list of subdirectories, each containing a trajectory.
        traj_dir = glob.glob(osp.join(dirname, "*"))
        traj_dir = sorted(traj_dir, key=lambda x: int(os.path.basename(x)))

        # Load the trajectories.
        trajectories = [
            Trajectory.load_from_folder(td, from_state=from_state) for td in traj_dir
        ]

        # Flatten all the trajectories into a single batch of transitions.
        keys = ["obs", "next_obs", "acts", "dones", "infos"]
        parts = {key: [] for key in keys}
        for traj in trajectories:
            parts["acts"].append(traj.acts)

            obs = traj.obs
            parts["obs"].append(obs[:-1])
            parts["next_obs"].append(obs[1:])

            dones = np.zeros(len(traj.acts), dtype=np.bool)
            dones[-1] = True
            parts["dones"].append(dones)

            if traj.infos is None:
                infos = np.array([{}] * len(traj))
            else:
                infos = traj.infos
            parts["infos"].append(infos)

        for key, part_list in parts.items():
            setattr(self, key, np.concatenate(part_list, axis=0))

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        obs = self.obs[idx]
        act = self.acts[idx]

        # Convert to tensors.
        obs = torch.FloatTensor(obs)
        act = torch.FloatTensor(act)

        return obs, act


class SequentialBCDataset(torch.utils.data.Dataset):
    """A dataset of trajectories for behavior cloning with sequential policies."""

    def __init__(
        self,
        dirname: str,
        from_state: bool,
        seq_len: int,
    ) -> None:
        # Get list of subdirectories, each containing a trajectory.
        traj_dir = glob.glob(osp.join(dirname, "*"))
        traj_dir = sorted(traj_dir, key=lambda x: int(os.path.basename(x)))

        # Load the trajectories.
        self.trajectories = [
            Trajectory.load_from_folder(td, from_state=from_state) for td in traj_dir
        ]

        self.seq_len = seq_len

    def __len__(self):
        return len(self.trajectories)

    def __getitem__(self, idx):
        traj = self.trajectories[idx]

        # Randomly sample a contiguous window within this sequence.
        seq_len = len(traj)
        start = random.randrange(seq_len - self.seq_len)
        end = start + self.seq_len
        obses = traj.obs[start:end]
        acts = traj.acts[start:end]

        assert len(obses) == len(acts)

        # Convert to tensors.
        obses = torch.FloatTensor(obses)
        acts = torch.FloatTensor(acts)

        return obses, acts
