"""
TRPO training loop.

Same step-based rollout structure as ppo/train.py; the only difference
is the update step (natural gradient + backtracking line search).
"""
from __future__ import annotations

import os
import random
import time
from collections import deque

import gymnasium as gym
import numpy as np
import torch

from . import ptu
from .config import parse_args, TRPOConfig
from .logger import Logger
from .rollout_buffer import RolloutBuffer
from .trpo_agent import TRPOAgent


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def env_spec(env: gym.Env):
    ob_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        return ob_dim, int(env.action_space.n), True
    elif isinstance(env.action_space, gym.spaces.Box):
        return ob_dim, int(np.prod(env.action_space.shape)), False
    else:
        raise NotImplementedError(f"Unsupported action space: {env.action_space}")


def maybe_clip_action(action, env: gym.Env, discrete: bool):
    if discrete:
        return action
    return np.clip(action, env.action_space.low, env.action_space.high)


def run(cfg: TRPOConfig) -> None:
    # -------------------- setup --------------------
    ptu.init_gpu(use_gpu=not cfg.no_gpu)
    set_seeds(cfg.seed)

    env = make_env(cfg.env_name, cfg.seed)
    ob_dim, ac_dim, discrete = env_spec(env)

    agent = TRPOAgent(
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        discrete=discrete,
        lr_value=cfg.lr_value,
        gamma=cfg.gamma,
        max_kl=cfg.max_kl,
        cg_steps=cfg.cg_steps,
        cg_damping=cfg.cg_damping,
        backtrack_steps=cfg.backtrack_steps,
        backtrack_coef=cfg.backtrack_coef,
        value_epochs=cfg.value_epochs,
        n_layers=cfg.n_layers,
        size=cfg.size,
    )

    buffer = RolloutBuffer(
        n_steps=cfg.n_steps,
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        discrete=discrete,
    )

    logger = Logger(cfg.log_dir)

    # rollout + update loop
    obs, _ = env.reset(seed=cfg.seed)
    episode_return = 0.0
    episode_length = 0
    recent_returns: deque = deque(maxlen=100)
    recent_lengths: deque = deque(maxlen=100)

    total_steps = 0
    updates = 0
    start_time = time.time()

    n_updates = cfg.total_timesteps // cfg.n_steps
    for update_i in range(n_updates):
        # 1. collect rollout
        for _ in range(cfg.n_steps):
            action, log_prob, value = agent.act(obs)
            env_action = maybe_clip_action(action, env, discrete)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated

            truncation_value = 0.0
            if truncated:
                truncation_value = agent.bootstrap_value(next_obs)

            buffer.add(
                obs=obs,
                action=action,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                value=value,
                log_prob=log_prob,
                truncation_value=truncation_value,
            )

            episode_return += float(reward)
            episode_length += 1
            total_steps += 1

            if done:
                recent_returns.append(episode_return)
                recent_lengths.append(episode_length)
                episode_return = 0.0
                episode_length = 0
                obs, _ = env.reset()
            else:
                obs = next_obs

        # 2. bootstrap final value, 3. compute returns and advantages
        last_value = agent.bootstrap_value(obs)
        buffer.compute_returns_and_advantages(last_value, cfg.gamma)

        # 4. TRPO update
        diagnostics = agent.update(buffer)
        updates += 1

        # 5. reset buffer
        buffer.reset()

        # logging
        sps = total_steps / max(1.0, time.time() - start_time)
        scalars = {
            "charts/env_steps": total_steps,
            "charts/updates": updates,
            "charts/steps_per_sec": sps,
            **diagnostics,
        }
        if recent_returns:
            scalars["charts/episodic_return_mean"] = float(np.mean(recent_returns))
            scalars["charts/episodic_length_mean"] = float(np.mean(recent_lengths))
        logger.log_scalars(scalars, total_steps)
        logger.flush()

        ep_ret = f"{np.mean(recent_returns):7.2f}" if recent_returns else "  n/a  "
        accepted = "✓" if diagnostics["trpo/step_accepted"] else "✗"
        print(
            f"[update {updates:4d} | steps {total_steps:8d} | sps {sps:6.0f}] "
            f"ep_return {ep_ret}  kl {diagnostics['trpo/approx_kl']:.4f}  "
            f"bt {diagnostics['trpo/backtrack_iters']:.0f}  {accepted}  "
            f"ev {diagnostics['trpo/explained_variance']:.3f}"
        )

    logger.close()
    env.close()

    torch.save(
        {"policy": agent.policy.state_dict(), "value_net": agent.value_net.state_dict()},
        os.path.join(cfg.log_dir, "model_final.pt"),
    )
    print(f"Saved final model to {os.path.join(cfg.log_dir, 'model_final.pt')}")


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
