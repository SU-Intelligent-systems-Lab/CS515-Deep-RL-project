"""
Actor-critic networks for PPO.

Separate policy and value MLPs (no shared trunk). Continuous actions use
a Normal with a state-independent learnable log_std (shape = [ac_dim]),
init = -0.5 so std starts around 0.6.
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
    is_policy_output: bool = False,
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

    # smaller init std on the policy head, larger on the value head
    final_std = 0.01 if is_policy_output else 1.0
    layers.append(layer_init(nn.Linear(in_size, output_size), std=final_std))
    layers.append(output_activation)
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    """Policy + value function (Categorical for discrete, Normal for continuous)."""

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

        if discrete:
            # logits over action set
            self.policy_net = build_mlp(ob_dim, ac_dim, n_layers, size, is_policy_output=True)
            self.log_std = None
        else:
            # mean per action dim; log_std is a free parameter shared across states
            self.policy_net = build_mlp(ob_dim, ac_dim, n_layers, size, is_policy_output=True)
            self.log_std = nn.Parameter(torch.full((ac_dim,), init_log_std, dtype=torch.float32))

        # scalar value head
        self.value_net = build_mlp(ob_dim, 1, n_layers, size)

    def _distribution(self, obs: torch.Tensor):
        if self.discrete:
            return Categorical(logits=self.policy_net(obs))
        mean = self.policy_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    @staticmethod
    def _log_prob(dist, action: torch.Tensor, discrete: bool) -> torch.Tensor:
        logp = dist.log_prob(action)
        if not discrete:
            # Normal returns per-dim log-probs; sum over independent action dims
            logp = logp.sum(-1)
        return logp

    @staticmethod
    def _entropy(dist, discrete: bool) -> torch.Tensor:
        ent = dist.entropy()
        if not discrete:
            ent = ent.sum(-1)
        return ent

    @torch.no_grad()
    def get_action(self, obs_np: np.ndarray) -> Tuple[np.ndarray, float, float]:
        """Sample an action for rollout. Returns (action, log_prob, value)."""
        obs = ptu.from_numpy(obs_np[None])
        dist = self._distribution(obs)
        action = dist.sample()
        logp = self._log_prob(dist, action, self.discrete)
        value = self.value_net(obs).squeeze(-1)

        action_np = ptu.to_numpy(action)[0]
        if self.discrete:
            return int(action_np), float(ptu.to_numpy(logp)[0]), float(ptu.to_numpy(value)[0])
        return action_np, float(ptu.to_numpy(logp)[0]), float(ptu.to_numpy(value)[0])

    @torch.no_grad()
    def get_value(self, obs_np: np.ndarray) -> float:
        """V(obs) for the rollout-end bootstrap."""
        obs = ptu.from_numpy(obs_np[None])
        value = self.value_net(obs).squeeze(-1)
        return float(ptu.to_numpy(value)[0])

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """With-grad forward. Returns (log_prob, entropy, value), each (B,)."""
        dist = self._distribution(obs)
        logp = self._log_prob(dist, actions, self.discrete)
        entropy = self._entropy(dist, self.discrete)
        value = self.value_net(obs).squeeze(-1)
        return logp, entropy, value
