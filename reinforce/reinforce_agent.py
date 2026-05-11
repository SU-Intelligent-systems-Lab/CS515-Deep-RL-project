"""
REINFORCEAgent: vanilla policy gradient with a learned value baseline.

For each batch of complete episodes:
    G_t        = reward-to-go from step t
    advantage  = G_t - V(s_t)
    pol_loss   = -mean(log pi(a|s) * advantage)
    value_loss = MSE(V(s), G_t)

No clipping, no trust region, no minibatch epochs. One gradient step per
batch, which keeps the variance high and learning slow.
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from . import ptu
from .networks import PolicyNet, ValueNet


class REINFORCEAgent:
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        lr: float = 3e-4,
        gamma: float = 0.99,
        entropy_coef: float = 0.01,
        n_layers: int = 2,
        size: int = 64,
    ) -> None:
        self.gamma = gamma
        self.entropy_coef = entropy_coef

        self.policy = PolicyNet(
            ob_dim=ob_dim,
            ac_dim=ac_dim,
            discrete=discrete,
            n_layers=n_layers,
            size=size,
        ).to(ptu.device)

        self.value = ValueNet(
            ob_dim=ob_dim,
            n_layers=n_layers,
            size=size,
        ).to(ptu.device)

        # separate optimizers: the policy step updates only the policy; the
        # value net is trained with a supervised MSE step.
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

        self.discrete = discrete

    def act(self, obs_np: np.ndarray):
        """Sample (action, log_prob) for one step during rollout."""
        return self.policy.act(obs_np)

    def update(self, trajectories: List[dict]) -> Dict[str, float]:
        """One gradient update over a batch of complete episodes.

        Each element of `trajectories` is a dict with keys
            obs       : (T, ob_dim) float32
            actions   : (T,) int64 (discrete) or (T, ac_dim) float32 (continuous)
            rewards   : list[float] of length T
            log_probs : (T,) float32  (stored at rollout time, recomputed here)
        """
        # 1. concatenate all episodes
        all_obs: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_returns: list[float] = []
        episode_returns: list[float] = []

        for traj in trajectories:
            rewards = traj["rewards"]
            T = len(rewards)

            # reward-to-go: G_t = r_t + gamma * G_{t+1}
            returns = np.zeros(T, dtype=np.float32)
            g = 0.0
            for t in reversed(range(T)):
                g = rewards[t] + self.gamma * g
                returns[t] = g

            all_obs.append(traj["obs"])
            all_actions.append(traj["actions"])
            all_returns.append(returns)
            episode_returns.append(float(np.sum(rewards)))

        obs_np = np.concatenate(all_obs, axis=0)
        returns_np = np.concatenate(all_returns, axis=0)
        # discrete actions: 1-D int64; continuous: 2-D float32
        actions_np = np.concatenate(all_actions, axis=0)

        # 2. move to tensors
        obs_t = ptu.from_numpy(obs_np)
        returns_t = ptu.from_numpy(returns_np)
        if self.discrete:
            actions_t = torch.from_numpy(actions_np).to(device=ptu.device)
        else:
            actions_t = ptu.from_numpy(actions_np)

        # 3. baseline and advantage (no grad through the baseline)
        with torch.no_grad():
            baseline = self.value(obs_t)
        advantages = returns_t - baseline
        # whitened advantages reduce gradient variance across the batch
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. policy gradient loss
        log_probs, entropy = self.policy.evaluate_actions(obs_t, actions_t)
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_mean = entropy.mean()
        total_policy_loss = policy_loss - self.entropy_coef * entropy_mean

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        # 5. value-net MSE step
        predicted_values = self.value(obs_t)
        value_loss = F.mse_loss(predicted_values, returns_t)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # 6. diagnostics
        var_returns = float(np.var(returns_np))
        var_residual = float(np.var(returns_np - ptu.to_numpy(baseline)))
        explained_var = 1.0 - var_residual / var_returns if var_returns > 0 else 0.0

        return {
            "losses/policy_loss": float(policy_loss.item()),
            "losses/value_loss": float(value_loss.item()),
            "losses/entropy": float(entropy_mean.item()),
            "reinforce/explained_variance": float(explained_var),
            "reinforce/mean_return": float(np.mean(episode_returns)),
            "reinforce/mean_length": float(np.mean([len(t["rewards"]) for t in trajectories])),
        }
