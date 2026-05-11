"""PPO configuration dataclass and CLI parser."""
import argparse
import os
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class PPOConfig:
    # Environment / experiment
    env_name: str
    exp_name: str
    seed: int
    total_timesteps: int

    # Rollout
    n_steps: int

    # Optimisation
    minibatch_size: int
    update_epochs: int
    lr: float
    gamma: float
    gae_lambda: float
    clip_eps: float
    target_kl: Optional[float]
    value_coef: float
    entropy_coef: float
    max_grad_norm: float

    # Network
    n_layers: int
    size: int

    # Infra
    log_dir: str
    no_gpu: bool


def parse_args() -> PPOConfig:
    p = argparse.ArgumentParser(description="PPO MVP trainer")

    # Environment / experiment
    p.add_argument("--env_name", type=str, default="CartPole-v1")
    p.add_argument("--exp_name", type=str, default="ppo")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total_timesteps", type=int, default=50_000)

    # Rollout
    p.add_argument("--n_steps", type=int, default=2048,
                   help="environment steps per on-policy rollout (and per update)")

    # Optimisation
    p.add_argument("--minibatch_size", type=int, default=64)
    p.add_argument("--update_epochs", type=int, default=10,
                   help="passes over each rollout during PPO update")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--target_kl", type=float, default=None,
                   help="KL early-stop threshold (disabled by default); set e.g. --target_kl 0.02 to enable")
    p.add_argument("--value_coef", type=float, default=0.5)
    p.add_argument("--entropy_coef", type=float, default=0.01,
                   help="use 0.0 for continuous-control envs like Pendulum")
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--gae_lambda", type=float, default=0.95,
                   help="GAE parameter lambda")


    # Network
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--size", type=int, default=64)

    # Infra
    p.add_argument("--log_dir", type=str, default=os.path.join("ppo", "data"))
    p.add_argument("--no_gpu", action="store_true",
                   help="force CPU even if CUDA is available")

    args = p.parse_args()

    # Build a timestamped run directory under log_dir.
    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.log_dir, f"{args.exp_name}_{args.env_name}_{ts}")

    return PPOConfig(
        env_name=args.env_name,
        exp_name=args.exp_name,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        minibatch_size=args.minibatch_size,
        update_epochs=args.update_epochs,
        lr=args.lr,
        gamma=args.gamma,
        clip_eps=args.clip_eps,
        target_kl=args.target_kl,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        n_layers=args.n_layers,
        size=args.size,
        log_dir=run_dir,
        gae_lambda=args.gae_lambda,
        no_gpu=args.no_gpu,
    )
