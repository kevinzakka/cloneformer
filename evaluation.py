"""Evaluate a trained policy in an environment."""

import collections
from typing import Sequence

import gym
import numpy as np
import torch
from statsmodels.stats.weightstats import DescrStatsW

EvalResult = collections.namedtuple(
    "EvalResult", ["mean_score", "ci95_lower", "ci95_upper", "std_score"]
)


class EvaluationProtocol:
    """Evaluate a trained policy on an env and calculate confidence intervals."""

    def __init__(self, policy: torch.nn.Module, n_rollouts: int) -> None:
        self.policy = policy
        self.n_rollouts = n_rollouts

    def obtain_scores(self, env: gym.Env) -> Sequence[float]:
        print(f"Performing {self.n_rollouts} evaluation rollouts.")
        eval_scores = []
        for _ in range(self.n_rollouts):
            env.seed()
            observation = env.reset()
            while True:
                with torch.no_grad():
                    action = self.policy.act(observation)
                observation, _, done, info = env.step(action)
                if done:
                    eval_scores.append(info["eval_score"])
                    break
        return eval_scores

    def do_eval(self, env: gym.Env) -> EvalResult:
        scores = self.obtain_scores(env)
        mean = np.mean(scores)
        interval = DescrStatsW(scores).tconfint_mean(0.05, "two-sided")
        std_dev = np.std(scores, ddof=1)
        return EvalResult(mean, interval[0], interval[1], std_dev)
