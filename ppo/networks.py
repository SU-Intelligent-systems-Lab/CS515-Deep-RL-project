"""
Networks for PPO: MLP builder + ActorCritic module.

Design notes:
 * Separate policy and value networks (not a shared trunk). For low-dim MLP envs
   this is simpler and — empirically — trains at least as well.
 * Continuous actions use a Normal distribution with a **state-independent**
   learnable log_std (shape = [ac_dim]). This is the standard PPO choice and
   avoids the instability of state-conditioned stds on simple envs.
 * log_std init = -0.5  ->  std ~= 0.6. Initialising at 0 (std=1) often causes
   divergence on Pendulum because initial actions are too large.
 * The Agent (not this module) owns the optimiser — one Adam over actor + critic
   + log_std.  That keeps the gradient step atomic.
"""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Normal

from . import ptu


Activation = Union[str, nn.Module]

_str_to_activation = {
    "relu": nn.ReLU(),
    "tanh": nn.Tanh(),
    "leaky_relu": nn.LeakyReLU(),
    "sigmoid": nn.Sigmoid(),
    "selu": nn.SELU(),
    "softplus": nn.Softplus(),
    "identity": nn.Identity(),
}

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Activation = "tanh",
    output_activation: Activation = "identity",
    is_policy_output: bool = False, # Add this flag
) -> nn.Sequential:
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers: list[nn.Module] = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(layer_init(nn.Linear(in_size, size)))
        layers.append(activation)
        in_size = size
        
    # Standard deviation for the final layer: 0.01 for policy, 1.0 for value
    final_std = 0.01 if is_policy_output else 1.0
    layers.append(layer_init(nn.Linear(in_size, output_size), std=final_std))
    layers.append(output_activation)
    return nn.Sequential(*layers)



class ActorCritic(nn.Module):
    """Policy + value function. Handles discrete (Categorical) and continuous (Normal)."""

    def __init__(
        self,
        ob_dim: int,
        ac_dim: int,
        discrete: bool,
        n_layers: int = 2,
        size: int = 64,
        init_log_std: float = -0.5,
    ) -> None:
        super().__init__()
        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        self.discrete = discrete

        # Policy head
        if discrete:
            # Output size = number of discrete actions (logits).
            self.policy_net = build_mlp(ob_dim, ac_dim, n_layers, size, is_policy_output=True)
            self.log_std = None
        else:
            # Output size = action dimension (mean of Gaussian per action dim).
            self.policy_net = build_mlp(ob_dim, ac_dim, n_layers, size, is_policy_output=True)
            # State-independent log_std, learned as free parameter.
            self.log_std = nn.Parameter(torch.full((ac_dim,), init_log_std, dtype=torch.float32))

        # Value head: always scalar.
        self.value_net = build_mlp(ob_dim, 1, n_layers, size)

    # ------------------------------------------------------------------ helpers

    def _distribution(self, obs: torch.Tensor):
        """Return the torch Distribution for these observations."""
        if self.discrete:
            logits = self.policy_net(obs)
            return Categorical(logits=logits)
        else:
            mean = self.policy_net(obs)
            std = torch.exp(self.log_std)  # broadcasts over batch dim
            return Normal(mean, std)

    @staticmethod
    def _log_prob(dist, action: torch.Tensor, discrete: bool) -> torch.Tensor:
        """Scalar log-prob per sample (summed over action dim for continuous)."""
        logp = dist.log_prob(action)
        if not discrete:
            # Normal.log_prob returns per-dim log-probs; actions are independent
            # across dims, so total log-prob = sum over dims.
            logp = logp.sum(-1)
        return logp

    @staticmethod
    def _entropy(dist, discrete: bool) -> torch.Tensor:
        """Scalar entropy per sample (summed over action dim for continuous)."""
        ent = dist.entropy()
        if not discrete:
            ent = ent.sum(-1)
        return ent

    # ------------------------------------------------------------------ rollout API

    @torch.no_grad()
    def get_action(self, obs_np: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Sample an action for rollout. Returns (action, log_prob, value) as numpy/scalars.

        Shape handling:
         * obs_np has shape (ob_dim,)  -> we add a batch dim for the forward pass
           and strip it before returning.
         * Discrete action returned as Python int (CartPole-style env.step wants int).
         * Continuous action returned as 1-D np.ndarray of shape (ac_dim,).
        """
        obs = ptu.from_numpy(obs_np[None])  # (1, ob_dim)
        dist = self._distribution(obs)
        action = dist.sample()               # (1,) discrete  or  (1, ac_dim) continuous
        logp = self._log_prob(dist, action, self.discrete)  # (1,)
        value = self.value_net(obs).squeeze(-1)              # (1,)

        action_np = ptu.to_numpy(action)[0]
        if self.discrete:
            # Categorical.sample returns int64 tensor; cast to Python int.
            return int(action_np), float(ptu.to_numpy(logp)[0]), float(ptu.to_numpy(value)[0])
        else:
            return action_np, float(ptu.to_numpy(logp)[0]), float(ptu.to_numpy(value)[0])

    @torch.no_grad()
    def get_value(self, obs_np: np.ndarray) -> float:
        """Value estimate for a single observation. Used for bootstrap at rollout end."""
        obs = ptu.from_numpy(obs_np[None])
        value = self.value_net(obs).squeeze(-1)
        return float(ptu.to_numpy(value)[0])

    # ------------------------------------------------------------------ update API

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass *with* grad. Returns (log_prob, entropy, value), each (B,).

        Called during the PPO update. `actions` must be shaped appropriately:
         * discrete  -> int64 tensor of shape (B,)
         * continuous -> float tensor of shape (B, ac_dim)
        """
        dist = self._distribution(obs)
        logp = self._log_prob(dist, actions, self.discrete)
        entropy = self._entropy(dist, self.discrete)
        value = self.value_net(obs).squeeze(-1)
        return logp, entropy, value
