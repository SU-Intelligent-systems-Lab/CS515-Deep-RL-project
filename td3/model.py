"""
Actor and (twin) Critic network definitions for TD3.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_mlp(input_size, output_size, n_layers, size,
              activation=nn.ReLU, output_activation=None):
    """Simple MLP builder """
    layers = []
    in_size = input_size
    for _ in range(n_layers):
        layers.append(nn.Linear(in_size, size))
        layers.append(activation())
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    if output_activation is not None:
        layers.append(output_activation())
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Deterministic policy pi_phi(s) -> a.

    Final tanh squashes actions to [-1, 1], then we rescale by max_action.
    """

    def __init__(self, ob_dim, ac_dim, max_action, n_layers=2, size=256):
        super(Actor, self).__init__()
        self.net = build_mlp(
            input_size=ob_dim,
            output_size=ac_dim,
            n_layers=n_layers,
            size=size,
            activation=nn.ReLU,
            output_activation=nn.Tanh,
        )
        self.max_action = max_action

    def forward(self, state):
        return self.max_action * self.net(state)


class Critic(nn.Module):
    """
    Twin critic: two independent Q-networks Q_theta1 and Q_theta2.

    Both share the same architecture but have independent parameters,
    initialized differently. TD3 uses min(Q1, Q2) as the target to
    combat overestimation bias (Clipped Double Q-learning).
    """

    def __init__(self, ob_dim, ac_dim, n_layers=2, size=256):
        super(Critic, self).__init__()

        # Q1 architecture
        self.q1 = build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        )

        # Q2 architecture (independent parameters)
        self.q2 = build_mlp(
            input_size=ob_dim + ac_dim,
            output_size=1,
            n_layers=n_layers,
            size=size,
        )

    def forward(self, state, action):
        """Returns both Q1(s,a) and Q2(s,a)."""
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

    def Q1(self, state, action):
        """Only Q1 is used for the actor update"""
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa)