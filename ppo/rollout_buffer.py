"""
Fixed-size on-policy rollout buffer for PPO.

Stores (obs, action, reward, terminated, truncated, value, log_prob) for
n_steps. After collection, compute_returns_and_advantages() fills in the
returns and GAE advantages; get_minibatches() yields shuffled minibatches.
"""
from __future__ import annotations

from typing import Iterator

import numpy as np
import torch

from . import ptu


class RolloutBuffer:
    def __init__(
        self,
        n_steps: int,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
    ) -> None:
        self.n_steps = n_steps
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.discrete = discrete

        self.obs = np.zeros((n_steps, ob_dim), dtype=np.float32)
        if discrete:
            # int64 matches torch's expected dtype for Categorical.log_prob
            self.actions = np.zeros((n_steps,), dtype=np.int64)
        else:
            self.actions = np.zeros((n_steps, ac_dim), dtype=np.float32)

        self.rewards = np.zeros((n_steps,), dtype=np.float32)

        # terminated vs truncated tracked separately: terminated -> next value = 0,
        # truncated (time-limit reached in a valid state) -> bootstrap with
        # V(s_at_truncation), stored in truncation_values below.
        self.terminated = np.zeros((n_steps,), dtype=np.float32)
        self.truncated = np.zeros((n_steps,), dtype=np.float32)
        self.truncation_values = np.zeros((n_steps,), dtype=np.float32)

        self.values = np.zeros((n_steps,), dtype=np.float32)
        self.log_probs = np.zeros((n_steps,), dtype=np.float32)

        # filled by compute_returns_and_advantages
        self.returns = np.zeros((n_steps,), dtype=np.float32)
        self.advantages = np.zeros((n_steps,), dtype=np.float32)

        self.ptr = 0

    # -------------------------------------------------------------- writes

    def add(
        self,
        obs: np.ndarray,
        action,  # int for discrete, np.ndarray for continuous
        reward: float,
        terminated: bool,
        truncated: bool,
        value: float,
        log_prob: float,
        truncation_value: float = 0.0,
    ) -> None:
        assert self.ptr < self.n_steps, "RolloutBuffer is full; call reset() after an update."
        i = self.ptr
        self.obs[i] = obs
        self.actions[i] = action
        self.rewards[i] = reward
        self.terminated[i] = 1.0 if terminated else 0.0
        self.truncated[i] = 1.0 if truncated else 0.0
        self.truncation_values[i] = truncation_value
        self.values[i] = value
        self.log_probs[i] = log_prob
        self.ptr += 1

    def is_full(self) -> bool:
        return self.ptr >= self.n_steps

    def reset(self) -> None:
        self.ptr = 0

    # -------------------------------------------------------------- post-processing

    def compute_returns_and_advantages(self, last_value: float, gamma: float, gae_lambda: float = 0.95) -> None:
        """GAE-lambda advantages computed backwards over the rollout."""
        assert self.is_full(), "Only call this at the end of a full rollout."

        last_gae_lam = 0.0
        for t in reversed(range(self.n_steps)):
            # value of the next state
            if self.terminated[t]:
                next_non_terminal = 0.0
                next_value = 0.0
            elif self.truncated[t]:
                next_non_terminal = 1.0
                next_value = self.truncation_values[t]
            else:
                next_non_terminal = 1.0
                next_value = self.values[t + 1] if t + 1 < self.n_steps else last_value

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            self.advantages[t] = last_gae_lam = delta + gamma * gae_lambda * next_non_terminal * last_gae_lam

        self.returns = self.advantages + self.values

    def get_minibatches(self, minibatch_size: int) -> Iterator[dict]:
        """Yield shuffled minibatches as tensors on ptu.device."""
        indices = np.random.permutation(self.n_steps)
        for start in range(0, self.n_steps, minibatch_size):
            mb = indices[start : start + minibatch_size]
            yield {
                "obs": ptu.from_numpy(self.obs[mb]),
                "actions": (
                    torch.from_numpy(self.actions[mb]).to(ptu.device)
                    if self.discrete
                    else ptu.from_numpy(self.actions[mb])
                ),
                "old_log_probs": ptu.from_numpy(self.log_probs[mb]),
                "advantages": ptu.from_numpy(self.advantages[mb]),
                "returns": ptu.from_numpy(self.returns[mb]),
            }