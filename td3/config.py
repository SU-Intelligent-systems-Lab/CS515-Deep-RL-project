"""
TD3 configuration dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TD3Config:
    # Environment / experiment
    env_name: str
    exp_name: str
    seed: int

    # Training schedule
    max_timesteps:   int
    start_timesteps: int      
    eval_freq:       int      
    eval_episodes:   int      

    # TD3 hyperparameters (paper defaults: tau=0.005, policy_freq=2, etc.)
    batch_size:   int
    discount:     float       # gamma
    tau:          float       # Polyak-averaging rate for target nets
    policy_noise: float       # std of target-smoothing noise 
    noise_clip:   float       # clip range for target-smoothing noise 
    policy_freq:  int         # delayed policy update frequency d
    expl_noise:   float       # exploration noise std (relative)

    # Network
    n_layers:      int
    size:          int
    learning_rate: float

    # Replay buffer
    buffer_size: int

    # Saving / loading
    save_model: bool
    load_model: str           

    # Rendering
    render:      bool
    render_freq: int

    # Plotting
    plot:      bool
    show_plot: bool

    # Infra
    log_dir: str
    no_gpu:  bool