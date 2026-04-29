"""
REINFORCEAgent — basic (vanilla) policy gradient with a learned value baseline.

Algorithm (Williams 1992 / Sutton & Barto):

    For each batch of complete episodes:
        G_t = sum_{t'=t}^{T} gamma^{t'-t} * r_{t'}     (reward-to-go)
        advantage_t = G_t - V(s_t)                       (baseline subtraction)
        policy_loss  = -mean(log π(a_t|s_t) * advantage_t)
        value_loss   = MSE(V(s_t), G_t)

Compared to the well-known PPO: no clipping, no trust region, no multiple epochs per
rollout. One gradient step per batch. Has high variance so slower learning.

Diagnostics returned each update:
 * losses/policy_loss   — mean PG loss
 * losses/value_loss    — MSE of value baseline
 * losses/entropy       — mean policy entropy
 * reinforce/explained_variance  — 1 - Var(G-V)/Var(G); tracks baseline fit
 * reinforce/mean_return         — mean episode return in this batch
 * reinforce/mean_length         — mean episode length in this batch
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

        # Separate optimizers: the policy gradient step updates only the policy.
        # The value network is updated independently with a supervised MSE step.
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)

        self.discrete = discrete

    # ------------------------------------------------------------------ rollout API

    def act(self, obs_np: np.ndarray):
        """Sample (action, log_prob) for one step. Called during episode collection."""
        return self.policy.act(obs_np)

    # ------------------------------------------------------------------ update API

    def update(self, trajectories: List[dict]) -> Dict[str, float]:
        """One gradient update from a batch of complete episodes.

        Each element of `trajectories` is a dict:
            obs       : np.ndarray (T, ob_dim)
            actions   : np.ndarray (T,) int64 or (T, ac_dim) float32
            rewards   : list[float] length T
            log_probs : np.ndarray (T,) — stored during rollout (not reused for update)

        Note: we RECOMPUTE log_probs during the update (evaluate_actions) rather than
        reusing the stored ones. REINFORCE does a single on-policy gradient step, so
        the stored and recomputed log-probs are identical — but recomputing keeps the
        computation graph intact for autograd.
        """
        # ---- 1. concatenate all episodes ----
        all_obs: list[np.ndarray] = []
        all_actions: list[np.ndarray] = []
        all_returns: list[float] = []
        episode_returns: list[float] = []

        for traj in trajectories:
            rewards = traj["rewards"]
            T = len(rewards)

            # Reward-to-go: G_t = r_t + gamma * G_{t+1}
            returns = np.zeros(T, dtype=np.float32)
            g = 0.0
            for t in reversed(range(T)):
                g = rewards[t] + self.gamma * g
                returns[t] = g

            all_obs.append(traj["obs"])        # (T, ob_dim)
            all_actions.append(traj["actions"])
            all_returns.append(returns)
            episode_returns.append(float(np.sum(rewards)))

        obs_np = np.concatenate(all_obs, axis=0)          # (N, ob_dim)
        returns_np = np.concatenate(all_returns, axis=0)   # (N,)

        # Actions can be either 1-D (discrete) or 2-D (continuous).
        actions_np = np.concatenate(all_actions, axis=0)

        # ---- 2. move to tensors ----
        obs_t = ptu.from_numpy(obs_np)                     # (N, ob_dim)
        returns_t = ptu.from_numpy(returns_np)             # (N,)
        if self.discrete:
            actions_t = torch.from_numpy(actions_np).to(device=ptu.device)  # int64
        else:
            actions_t = ptu.from_numpy(actions_np)         # (N, ac_dim) float32

        # ---- 3. compute baseline and advantage ----
        # Baseline evaluation under no_grad — we update the value net separately.
        with torch.no_grad():
            baseline = self.value(obs_t)                   # (N,)

        advantages = returns_t - baseline                  # (N,)

        # Advantage normalisation: reduces variance across the batch, similar to
        # the per-minibatch normalisation in PPO.
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---- 4. policy gradient loss ----
        log_probs, entropy = self.policy.evaluate_actions(obs_t, actions_t)
        policy_loss = -(log_probs * advantages.detach()).mean()
        entropy_mean = entropy.mean()
        total_policy_loss = policy_loss - self.entropy_coef * entropy_mean

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        # ---- 5. value (baseline) MSE loss ----
        predicted_values = self.value(obs_t)               # (N,) with grad
        value_loss = F.mse_loss(predicted_values, returns_t)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # ---- 6. diagnostics ----
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
