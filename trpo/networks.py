"""
Networks for TRPO: separate PolicyNet and ValueNet (no shared trunk).

PolicyNet exposes distribution() so the agent can compute KL divergences
and Fisher-vector products directly on the action distribution.
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


def build_mlp(
    input_size: int,
    output_size: int,
    n_layers: int,
    size: int,
    activation: Activation = "tanh",
    output_activation: Activation = "identity",
) -> nn.Sequential:
    """Stack of Linear + activation layers."""
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers: list[nn.Module] = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    layers.append(output_activation)
    return nn.Sequential(*layers)


class PolicyNet(nn.Module):
    """Stochastic policy; the public distribution() is used for KL and FVP."""

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

        self.net = build_mlp(ob_dim, ac_dim, n_layers, size)
        if not discrete:
            self.log_std = nn.Parameter(
                torch.full((ac_dim,), init_log_std, dtype=torch.float32)
            )
        else:
            self.log_std = None

    def distribution(self, obs: torch.Tensor):
        """Distribution over actions for the given batch of obs."""
        if self.discrete:
            return Categorical(logits=self.net(obs))
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray) -> Tuple[object, float]:
        """Sample (action, log_prob) for one observation."""
        obs = ptu.from_numpy(obs_np[None])
        dist = self.distribution(obs)
        action = dist.sample()
        logp = dist.log_prob(action)
        if not self.discrete:
            logp = logp.sum(-1)
        action_np = ptu.to_numpy(action)[0]
        if self.discrete:
            return int(action_np), float(ptu.to_numpy(logp)[0])
        return action_np, float(ptu.to_numpy(logp)[0])

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """With-grad (log_prob, entropy) on the given batch."""
        dist = self.distribution(obs)
        logp = dist.log_prob(actions)
        if not self.discrete:
            logp = logp.sum(-1)
        entropy = dist.entropy()
        if not self.discrete:
            entropy = entropy.sum(-1)
        return logp, entropy


class ValueNet(nn.Module):
    """V(s) trained separately with Adam + MSE."""

    def __init__(self, ob_dim: int, n_layers: int = 2, size: int = 64) -> None:
        super().__init__()
        self.net = build_mlp(ob_dim, 1, n_layers, size)

    @torch.no_grad()
    def get_value(self, obs_np: np.ndarray) -> float:
        obs = ptu.from_numpy(obs_np[None])
        return float(ptu.to_numpy(self.net(obs).squeeze(-1))[0])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)
