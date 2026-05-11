"""
Replay a trained REINFORCE policy live and report evaluation statistics.

Loads a model_final.pt checkpoint, rebuilds the env with render_mode="human",
plays --episodes episodes, prints per-episode returns and a final mean/std.
"""
from __future__ import annotations

import argparse
import time

import gymnasium as gym
import numpy as np
import torch

from . import ptu
from .networks import PolicyNet


def env_spec(env: gym.Env):
    """Return (ob_dim, ac_dim, discrete) from an env's spaces."""
    ob_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        return ob_dim, int(env.action_space.n), True
    elif isinstance(env.action_space, gym.spaces.Box):
        return ob_dim, int(np.prod(env.action_space.shape)), False
    else:
        raise NotImplementedError(f"Unsupported action space: {env.action_space}")


@torch.no_grad()
def select_action(policy: PolicyNet, obs_np: np.ndarray, deterministic: bool):
    """Pick an action for rendering.

    When deterministic=True, use the mean (continuous) or argmax (discrete).
    Otherwise sample from the stochastic policy, matching training behaviour.
    """
    if not deterministic:
        action, _ = policy.act(obs_np)
        return action

    obs = ptu.from_numpy(obs_np[None])
    dist = policy._distribution(obs)
    if policy.discrete:
        action_t = dist.probs.argmax(dim=-1)
        return int(ptu.to_numpy(action_t)[0])
    else:
        mean_t = dist.mean
        return ptu.to_numpy(mean_t)[0]


def run(args: argparse.Namespace) -> None:
    ptu.init_gpu(use_gpu=False)  # rendering is CPU-only; avoids CUDA overhead

    env = gym.make(args.env_name, render_mode="human")
    ob_dim, ac_dim, discrete = env_spec(env)

    policy = PolicyNet(
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        discrete=discrete,
        n_layers=args.n_layers,
        size=args.size,
    ).to(ptu.device)

    state = torch.load(args.checkpoint, map_location=ptu.device)
    policy.load_state_dict(state["policy"])
    policy.eval()

    returns: list[float] = []
    lengths: list[int] = []

    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        ep_return = 0.0
        ep_length = 0
        while True:
            action = select_action(policy, obs, deterministic=args.deterministic)
            if not discrete:
                action = np.clip(action, env.action_space.low, env.action_space.high)
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += float(reward)
            ep_length += 1
            if args.fps > 0:
                time.sleep(1.0 / args.fps)
            if terminated or truncated:
                break
        returns.append(ep_return)
        lengths.append(ep_length)
        print(f"[episode {ep + 1:3d}]  return {ep_return:8.2f}  length {ep_length:4d}")

    env.close()

    mean_ret = float(np.mean(returns))
    std_ret = float(np.std(returns))
    print("-" * 50)
    print(f"Evaluation over {args.episodes} episodes:")
    print(f"  return  : {mean_ret:.2f} ± {std_ret:.2f}")
    print(f"  length  : {float(np.mean(lengths)):.1f} ± {float(np.std(lengths)):.1f}")


def main() -> None:
    p = argparse.ArgumentParser(description="Render a trained REINFORCE policy live.")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="path to model_final.pt saved by reinforce.train")
    p.add_argument("--env_name", type=str, required=True,
                   help="must match the env the policy was trained on")
    p.add_argument("--episodes", type=int, default=5,
                   help="number of episodes to play")
    p.add_argument("--seed", type=int, default=1000,
                   help="base seed for env.reset (offset by episode index)")
    p.add_argument("--deterministic", action="store_true",
                   help="use the policy mean/argmax instead of sampling")
    p.add_argument("--fps", type=float, default=0.0,
                   help="cap render speed (frames per second); 0 = uncapped")

    # Must match the architecture used during training. Defaults match config.py.
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--size", type=int, default=64)

    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
