"""TRPO configuration dataclass and CLI parser."""
import argparse
import os
import time
from dataclasses import dataclass


@dataclass
class TRPOConfig:
    # Environment / experiment
    env_name: str
    exp_name: str
    seed: int
    total_timesteps: int

    # Rollout
    n_steps: int

    # TRPO-specific
    max_kl: float
    cg_steps: int
    cg_damping: float
    backtrack_steps: int
    backtrack_coef: float

    # Value net (updated separately with Adam)
    lr_value: float
    value_epochs: int
    gamma: float

    # Network
    n_layers: int
    size: int

    # Infra
    log_dir: str
    no_gpu: bool


def parse_args() -> TRPOConfig:
    p = argparse.ArgumentParser(description="TRPO trainer")

    # Environment / experiment
    p.add_argument("--env_name", type=str, default="CartPole-v1")
    p.add_argument("--exp_name", type=str, default="trpo")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total_timesteps", type=int, default=100_000,
                   help="increase to 1_000_000 for Pendulum-v1")

    # Rollout
    p.add_argument("--n_steps", type=int, default=2048,
                   help="environment steps per on-policy rollout")

    # TRPO
    p.add_argument("--max_kl", type=float, default=0.01,
                   help="trust region radius δ (KL constraint)")
    p.add_argument("--cg_steps", type=int, default=10,
                   help="conjugate gradient iterations")
    p.add_argument("--cg_damping", type=float, default=0.1,
                   help="Fisher damping for numerical stability")
    p.add_argument("--backtrack_steps", type=int, default=10,
                   help="maximum line search backtracks")
    p.add_argument("--backtrack_coef", type=float, default=0.5,
                   help="line search step reduction factor")

    # Value
    p.add_argument("--lr_value", type=float, default=1e-3)
    p.add_argument("--value_epochs", type=int, default=5,
                   help="Adam steps on value net per rollout")
    p.add_argument("--gamma", type=float, default=0.99)

    # Network
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--size", type=int, default=64)

    # Infra
    p.add_argument("--log_dir", type=str, default=os.path.join("trpo", "data"))
    p.add_argument("--no_gpu", action="store_true",
                   help="force CPU even if CUDA is available")

    args = p.parse_args()

    ts = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(args.log_dir, f"{args.exp_name}_{args.env_name}_{ts}")

    return TRPOConfig(
        env_name=args.env_name,
        exp_name=args.exp_name,
        seed=args.seed,
        total_timesteps=args.total_timesteps,
        n_steps=args.n_steps,
        max_kl=args.max_kl,
        cg_steps=args.cg_steps,
        cg_damping=args.cg_damping,
        backtrack_steps=args.backtrack_steps,
        backtrack_coef=args.backtrack_coef,
        lr_value=args.lr_value,
        value_epochs=args.value_epochs,
        gamma=args.gamma,
        n_layers=args.n_layers,
        size=args.size,
        log_dir=run_dir,
        no_gpu=args.no_gpu,
    )
