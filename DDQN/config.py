"""
Double DQN configuration dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class DDQNConfig:
    # Environment / experiment
    env_name: str
    exp_name: str
    seed: int

    # Training schedule
    n_episodes: int
    max_t:      int           # max steps per episode

    # Epsilon-greedy schedule
    eps_start: float
    eps_end:   float
    eps_decay: float          # multiplicative decay per episode

    # Algorithm switch
    use_ddqn: bool            # True = Double DQN, False = vanilla DQN

    # Optional early-stop (mean-of-100 episode score)
    score_threshold: Optional[float]

    # Plotting
    plot:      bool
    show_plot: bool

    # Infra
    log_dir: str
    no_gpu:  bool