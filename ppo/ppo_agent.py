"""
PPOAgent: clipped-surrogate update with optional KL early stopping.

Loss per minibatch:
    L = -clipped_surrogate + value_coef * value_MSE - entropy_coef * entropy
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import ptu
from .networks import ActorCritic
from .rollout_buffer import RolloutBuffer


class PPOAgent:
    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        minibatch_size: int = 64,
        target_kl: Optional[float] = None,
        n_layers: int = 2,
        size: int = 64,
    ) -> None:
        self.net = ActorCritic(
            ob_dim=ob_dim,
            ac_dim=ac_dim,
            discrete=discrete,
            n_layers=n_layers,
            size=size,
        ).to(ptu.device)

        # One Adam over all parameters: policy, value, and log_std for continuous.
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size
        self.target_kl = target_kl

    # ------------------------------------------------------------------ rollout API

    def act(self, obs_np: np.ndarray):
        """Sample (action, log_prob, value) for one observation."""
        return self.net.get_action(obs_np)

    def bootstrap_value(self, obs_np: np.ndarray) -> float:
        """V(obs) for the rollout-end bootstrap."""
        return self.net.get_value(obs_np)

    # ------------------------------------------------------------------ update API













    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """K epochs of minibatch updates on `buffer`. Returns averaged diagnostics."""
        b_advantages = buffer.advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        buffer.advantages = b_advantages

        # explained variance, computed once per buffer
        returns_all = buffer.returns.copy()
        values_all = buffer.values.copy()
        var_returns = np.var(returns_all)
        explained_var = (
            1.0 - np.var(returns_all - values_all) / var_returns
            if var_returns > 0
            else 0.0
        )

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropies: list[float] = []
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        epochs_completed = 0
        kl_exceeded = False

        for _ in range(self.update_epochs):
            if kl_exceeded:
                break
            epochs_completed += 1
            epoch_kls: list[float] = []
            for mb in buffer.get_minibatches(self.minibatch_size):
                obs = mb["obs"]
                actions = mb["actions"]
                old_log_probs = mb["old_log_probs"]
                advantages = mb["advantages"]
                returns = mb["returns"]

                advantages = mb["advantages"]

                new_log_probs, entropy, value = self.net.evaluate_actions(obs, actions)

                # importance ratio = exp(log_pi_new - log_pi_old)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # clipped surrogate
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # plain MSE on the value head (no value clipping)
                value_loss = F.mse_loss(value, returns)

                entropy_mean = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy_mean
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # cheap diagnostics on the same minibatch
                with torch.no_grad():
                    approx_kl = 0.5 * ((old_log_probs - new_log_probs) ** 2).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_mean.item())
                approx_kls.append(approx_kl)
                epoch_kls.append(approx_kl)
                clip_fractions.append(clip_frac)

            # optional KL early stop, gated on the mean KL of the epoch
            # (per-minibatch checking is too noisy)
            if self.target_kl is not None and epoch_kls:
                mean_epoch_kl = float(np.mean(epoch_kls))
                if mean_epoch_kl > self.target_kl:
                    kl_exceeded = True

        return {
            "losses/policy_loss": float(np.mean(policy_losses)),
            "losses/value_loss": float(np.mean(value_losses)),
            "losses/entropy": float(np.mean(entropies)),
            "ppo/approx_kl": float(np.mean(approx_kls)),
            "ppo/clip_fraction": float(np.mean(clip_fractions)),
            "ppo/explained_variance": float(explained_var),
            "ppo/epochs_completed": float(epochs_completed),
        }
