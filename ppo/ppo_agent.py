"""
PPOAgent — the core PPO update.

The objective being minimised each minibatch step is:

    L = -clipped_surrogate + value_coef * value_MSE - entropy_coef * entropy

with clipped_surrogate = min(ratio * A, clip(ratio, 1-e, 1+e) * A) and
ratio = exp(new_log_prob - old_log_prob). `old_log_prob` comes straight from
the rollout (detached by construction — numpy in the buffer).

Diagnostics returned each update (for TensorBoard):
 * approx_kl         ~ 0.5 * E[(old_logp - new_logp)^2]   (Schulman's k3 estimator)
 * clip_fraction     fraction of samples where |ratio - 1| > clip_eps
 * explained_variance  1 - Var(returns - values) / Var(returns); monitors critic fit
 * epochs_completed  how many of the K epochs actually ran (< K when KL early-stop fired)
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

        # Single optimiser over everything — policy, value, and (for continuous) log_std.
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
        """Return (action, log_prob, value) for a single observation."""
        return self.net.get_action(obs_np)

    def bootstrap_value(self, obs_np: np.ndarray) -> float:
        """V(obs) — used to bootstrap the final return in the rollout buffer."""
        return self.net.get_value(obs_np)

    # ------------------------------------------------------------------ update API













    def update(self, buffer: RolloutBuffer) -> Dict[str, float]:
        """Run K epochs of minibatch PPO updates over `buffer`. Returns averaged diagnostics."""
        b_advantages = buffer.advantages
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)
        buffer.advantages = b_advantages
        # For explained variance we need per-buffer variances, computed once.
        returns_all = buffer.returns.copy()
        values_all = buffer.values.copy()
        var_returns = np.var(returns_all)
        explained_var = (
            1.0 - np.var(returns_all - values_all) / var_returns
            if var_returns > 0
            else 0.0
        )

        # Accumulate diagnostics.
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

                # Per-minibatch advantage normalization — the standard PPO choice.
                # Using std() + 1e-8 to guard against the degenerate |advantage| = 0 case.
                advantages = mb["advantages"]

                #advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                new_log_probs, entropy, value = self.net.evaluate_actions(obs, actions)

                # ratio = pi_new / pi_old = exp(log_pi_new - log_pi_old)
                ratio = torch.exp(new_log_probs - old_log_probs)

                # Clipped surrogate.
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss. MVP: plain MSE (no value clipping).
                value_loss = F.mse_loss(value, returns)

                # Entropy bonus (negative because we subtract it in the loss).
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

                # Diagnostics (cheap, done on the same minibatch).
                with torch.no_grad():
                    # Schulman's k3 approximate KL: 0.5 * E[(old - new)^2]. Cheap and unbiased.
                    approx_kl = 0.5 * ((old_log_probs - new_log_probs) ** 2).mean().item()
                    clip_frac = ((ratio - 1.0).abs() > self.clip_eps).float().mean().item()

                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy_mean.item())
                approx_kls.append(approx_kl)
                epoch_kls.append(approx_kl)
                clip_fractions.append(clip_frac)

            # KL early stopping (opt-in — only active when target_kl is set).
            # We check the MEAN KL over the epoch (not per-minibatch) — matches
            # Per-minibatch checking is too noisy and stops far too aggressively.
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
