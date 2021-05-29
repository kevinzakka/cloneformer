import collections

import gym
import numpy as np


class FrameStack(gym.Wrapper):
    """Stack the last k frames of the env into a flat array.

    This is useful for allowing the RL policy to infer temporal information.
    """

    def __init__(self, env, k):
        """Constructor.

        Args:
            env: A gym env.
            k: The number of frames to stack.
        """
        super().__init__(env)

        assert isinstance(k, int), "k must be an integer."

        self._k = k
        self._frames = collections.deque([], maxlen=k)

        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype,
        )

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


class ActionRepeat(gym.Wrapper):
    """Repeat the agent's action N times in the environment."""

    def __init__(self, env, repeat):
        """Constructor.

        Args:
            env: A gym env.
            repeat: The number of times to repeat the action per single underlying env
                step.
        """
        super().__init__(env)

        assert repeat > 1, "repeat should be greater than 1."
        self._repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._repeat):
            obs, rew, done, info = self.env.step(action)
            total_reward += rew
            if done:
                break
        return obs, total_reward, done, info


def wrapper_from_config(config, env):
    """Wrap the environment based on values in the config."""
    if config.action_repeat > 1:
        env = ActionRepeat(env, config.action_repeat)
    if config.frame_stack > 1:
        env = FrameStack(env, config.frame_stack)
    return env
