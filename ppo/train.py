"""
PPO training loop.

Structure (the canonical on-policy loop):
  while total_steps < budget:
      1. Collect n_steps of rollouts from a single env (store obs, action,
         reward, done, value, log_prob).
      2. Bootstrap the final value V(s_last).
      3. Compute returns + advantages over the buffer.
      4. Run K epochs of minibatch PPO updates.
      5. Reset the buffer.

Uses `gymnasium` (not the deprecated `gym`), so env.step returns a 5-tuple:
    obs, reward, terminated, truncated, info

Episode returns/lengths are tracked in-loop and logged when an episode ends.
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
from .config import parse_args, PPOConfig
from .logger import Logger
from .ppo_agent import PPOAgent
from .rollout_buffer import RolloutBuffer


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_name: str, seed: int) -> gym.Env:
    env = gym.make(env_name)
    # Seed once at creation. env.reset(seed=...) only on the very first reset,
    # after that use unseeded reset() to preserve exploration variety.
    env.action_space.seed(seed)
    env.observation_space.seed(seed)
    return env


def env_spec(env: gym.Env):
    """Return (ob_dim, ac_dim, discrete) from an env's spaces."""
    ob_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        return ob_dim, int(env.action_space.n), True
    elif isinstance(env.action_space, gym.spaces.Box):
        return ob_dim, int(np.prod(env.action_space.shape)), False
    else:
        raise NotImplementedError(f"Unsupported action space: {env.action_space}")


def maybe_clip_action(action, env: gym.Env, discrete: bool):
    """For continuous envs, clip to the action_space bounds before env.step.

    Important: clip for the env, but keep the UN-CLIPPED action in the buffer
    (its log-prob is what we stored). Clipping only the env-facing action
    preserves the importance ratio — clipping the stored action would drift it.
    """
    if discrete:
        return action
    return np.clip(action, env.action_space.low, env.action_space.high)


def run(cfg: PPOConfig) -> None:
    # -------------------- setup --------------------
    ptu.init_gpu(use_gpu=not cfg.no_gpu)
    set_seeds(cfg.seed)

    env = make_env(cfg.env_name, cfg.seed)
    ob_dim, ac_dim, discrete = env_spec(env)

    agent = PPOAgent(
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        discrete=discrete,
        lr=cfg.lr,
        clip_eps=cfg.clip_eps,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        max_grad_norm=cfg.max_grad_norm,
        update_epochs=cfg.update_epochs,
        minibatch_size=cfg.minibatch_size,
        target_kl=cfg.target_kl,
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

    # -------------------- rollout + update loop --------------------
    obs, _ = env.reset(seed=cfg.seed)  # seed ONLY on the very first reset
    episode_return = 0.0
    episode_length = 0
    # Keep the last 100 episode returns for a smooth running mean.
    recent_returns: deque = deque(maxlen=100)
    recent_lengths: deque = deque(maxlen=100)

    total_steps = 0
    updates = 0
    start_time = time.time()

    n_updates = cfg.total_timesteps // cfg.n_steps
    for update_i in range(n_updates):
        # --------------- 1. collect rollout ---------------
        for _ in range(cfg.n_steps):
            action, log_prob, value = agent.act(obs)
            env_action = maybe_clip_action(action, env, discrete)
            next_obs, reward, terminated, truncated, _ = env.step(env_action)
            done = terminated or truncated

            truncation_value = 0.0
            if truncated:
                # Bootstrap the value of the actual state before auto-reset
                truncation_value = agent.bootstrap_value(next_obs)

            buffer.add(
                obs=obs,
                action=action,
                reward=float(reward),
                terminated=terminated,
                truncated=truncated,
                value=value,
                log_prob=log_prob,
                truncation_value=truncation_value
            )
            
            episode_return += float(reward)
            episode_length += 1
            total_steps += 1

            if done:
                recent_returns.append(episode_return)
                recent_lengths.append(episode_length)
                episode_return = 0.0
                episode_length = 0
                obs, _ = env.reset()  # no seed — preserve exploration variety
            else:
                obs = next_obs

        # --------------- 2. bootstrap + 3. returns/advantages ---------------
        last_value = agent.bootstrap_value(obs)
        buffer.compute_returns_and_advantages(last_value, cfg.gamma, cfg.gae_lambda)

        # --------------- 4. PPO update ---------------
        diagnostics = agent.update(buffer)
        updates += 1

        # --------------- 5. reset buffer ---------------
        buffer.reset()

        # --------------- logging ---------------
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

        # A compact progress print every update; cheap and helps CLI debugging.
        ep_ret = f"{np.mean(recent_returns):7.2f}" if recent_returns else "  n/a  "
        print(
            f"[update {updates:4d} | steps {total_steps:8d} | sps {sps:6.0f}] "
            f"ep_return {ep_ret}  kl {diagnostics['ppo/approx_kl']:.4f}  "
            f"clip {diagnostics['ppo/clip_fraction']:.3f}  "
            f"ev {diagnostics['ppo/explained_variance']:.3f}  "
            f"epochs {diagnostics['ppo/epochs_completed']:.0f}/{cfg.update_epochs}"
        )

    logger.close()
    env.close()

    torch.save(agent.net.state_dict(), os.path.join(cfg.log_dir, "model_final.pt"))
    print(f"Saved final model to {os.path.join(cfg.log_dir, 'model_final.pt')}")


def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()