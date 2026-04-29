"""
Command-line argument parsing for REINFORCE training.

No clip_eps, no value_coef, no target_kl, no
minibatch_size, no update_epochs. One gradient step per episode batch.
"""
import argparse
import os
import time
from dataclasses import dataclass


@dataclass
class REINFORCEConfig:
    # Environment / experiment
    env_name: str
    exp_name: str
    seed: int
    total_episodes: int

    # Rollout
    min_episodes_per_update: int   # collect at least this many episodes before each gradient step

    # Optimisation
    lr: float
    gamma: float
    entropy_coef: float

    # Network
    n_layers: int
    size: int

    # Infra
    log_dir: str
    no_gpu: bool


def parse_args() -> REINFORCEConfig:
    p = argparse.ArgumentParser(description="REINFORCE trainer")

    # Environment / experiment
    p.add_argument("--env_name", type=str, default="CartPole-v1")
    p.add_argument("--exp_name", type=str, default="reinforce")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total_episodes", type=int, default=500,
                   help="total training episodes (increase to ~5000 for Pendulum)")

    # Rollout
    p.add_argument("--min_episodes_per_update", type=int, default=10,
                   help="episodes to collect before each gradient step")

    # Optimisation
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--entropy_coef", type=float, default=0.01,
                   help="use 0.0 for continuous-control envs like Pendulum")

    # Network
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--size", type=int, default=64)

    # Infra
    p.add_argument("--log_dir", type=str, default=os.path.join("reinforce", "data"))
    p.add_argument("--no_gpu", action="store_true",
                   help="force CPU even if CUDA is available")

    args = p.parse_args()

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.log_dir, f"{args.exp_name}_{args.env_name}_{ts}")

    return REINFORCEConfig(
        env_name=args.env_name,
        exp_name=args.exp_name,
        seed=args.seed,
        total_episodes=args.total_episodes,
        min_episodes_per_update=args.min_episodes_per_update,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy_coef,
        n_layers=args.n_layers,
        size=args.size,
        log_dir=run_dir,
        no_gpu=args.no_gpu,
    )
