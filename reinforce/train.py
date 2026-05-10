"""
REINFORCE training loop.

Structure (episodic on-policy loop):
  while total_episodes < budget:
      1. Collect min_episodes_per_update complete episodes.
         For each step: obs -> act -> store (obs, action, reward, log_prob).
         A truncated episode counts as complete.
      2. Compute reward-to-go G_t for every step in every episode.
      3. One gradient step on policy + value baseline.
      4. Log diagnostics.

Uses `gymnasium`, so env.step returns a 5-tuple:
    obs, reward, terminated, truncated, info
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
from .config import parse_args, REINFORCEConfig
from .logger import Logger
from .reinforce_agent import REINFORCEAgent


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
    """Return (ob_dim, ac_dim, discrete) from an env's spaces."""
    ob_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        return ob_dim, int(env.action_space.n), True
    elif isinstance(env.action_space, gym.spaces.Box):
        return ob_dim, int(np.prod(env.action_space.shape)), False
    else:
        raise NotImplementedError(f"Unsupported action space: {env.action_space}")


def maybe_clip_action(action, env: gym.Env, discrete: bool):
    """Clip continuous actions to env bounds before env.step (log_prob stays correct)."""
    if discrete:
        return action
    return np.clip(action, env.action_space.low, env.action_space.high)


def collect_episode(env: gym.Env, agent: REINFORCEAgent, discrete: bool) -> dict:
    """Run one complete episode and return a trajectory dict.

    Returns:
        obs       : np.ndarray (T, ob_dim)
        actions   : np.ndarray (T,) int64  or  (T, ac_dim) float32
        rewards   : list[float] length T
        log_probs : np.ndarray (T,)  [stored but not used in update — for diagnostics]
    """
    obs, _ = env.reset()
    obs_list: list[np.ndarray] = []
    action_list: list = []
    reward_list: list[float] = []
    logp_list: list[float] = []

    while True:
        action, log_prob = agent.act(obs)
        env_action = maybe_clip_action(action, env, discrete)
        next_obs, reward, terminated, truncated, _ = env.step(env_action)

        obs_list.append(obs.copy())
        action_list.append(action)
        reward_list.append(float(reward))
        logp_list.append(log_prob)

        if terminated or truncated:
            break
        obs = next_obs

    obs_arr = np.stack(obs_list, axis=0).astype(np.float32)       # (T, ob_dim)
    if discrete:
        actions_arr = np.array(action_list, dtype=np.int64)        # (T,)
    else:
        actions_arr = np.stack(action_list, axis=0).astype(np.float32)  # (T, ac_dim)

    return {
        "obs": obs_arr,
        "actions": actions_arr,
        "rewards": reward_list,
        "log_probs": np.array(logp_list, dtype=np.float32),
    }


def run(cfg: REINFORCEConfig) -> None:
    # -------------------- setup --------------------
    ptu.init_gpu(use_gpu=not cfg.no_gpu)
    set_seeds(cfg.seed)

    env = make_env(cfg.env_name, cfg.seed)
    ob_dim, ac_dim, discrete = env_spec(env)

    agent = REINFORCEAgent(
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        discrete=discrete,
        lr=cfg.lr,
        gamma=cfg.gamma,
        entropy_coef=cfg.entropy_coef,
        n_layers=cfg.n_layers,
        size=cfg.size,
    )

    logger = Logger(cfg.log_dir)

    # -------------------- episode loop --------------------
    total_episodes = 0
    total_steps = 0
    updates = 0
    start_time = time.time()
    recent_returns: deque = deque(maxlen=100)
    recent_lengths: deque = deque(maxlen=100)

    while total_episodes < cfg.total_episodes:
        # --------------- 1. collect batch of episodes ---------------
        trajectories = []
        for _ in range(cfg.min_episodes_per_update):
            traj = collect_episode(env, agent, discrete)
            trajectories.append(traj)
            total_episodes += 1
            total_steps += len(traj["rewards"])
            ep_ret = float(np.sum(traj["rewards"]))
            recent_returns.append(ep_ret)
            recent_lengths.append(len(traj["rewards"]))

            # Stop collecting if we've hit the total budget.
            if total_episodes >= cfg.total_episodes:
                break

        # --------------- 2+3. compute returns + gradient step ---------------
        diagnostics = agent.update(trajectories)
        updates += 1

        # --------------- logging ---------------
        sps = total_steps / max(1.0, time.time() - start_time)
        scalars = {
            "charts/env_steps": total_steps,
            "charts/total_episodes": total_episodes,
            "charts/updates": updates,
            "charts/steps_per_sec": sps,
            **diagnostics,
        }
        if recent_returns:
            scalars["charts/episodic_return_mean"] = float(np.mean(recent_returns))
            scalars["charts/episodic_length_mean"] = float(np.mean(recent_lengths))
        logger.log_scalars(scalars, total_steps)
        logger.flush()

        ep_ret_str = f"{np.mean(recent_returns):7.2f}" if recent_returns else "  n/a  "
        print(
            f"[update {updates:4d} | ep {total_episodes:6d} | steps {total_steps:8d} | sps {sps:6.0f}] "
            f"ep_return {ep_ret_str}  "
            f"pol_loss {diagnostics['losses/policy_loss']:7.4f}  "
            f"val_loss {diagnostics['losses/value_loss']:7.4f}  "
            f"ev {diagnostics['reinforce/explained_variance']:.3f}"
        )

    logger.close()
    env.close()

    torch.save(
        {"policy": agent.policy.state_dict(), "value": agent.value.state_dict()},
        os.path.join(cfg.log_dir, "model_final.pt"),
    )
    print("\n" + "=" * 60)
    print("Training complete. To watch the trained agent play:")
    print(f'  python main.py --mode test --load_dir "{cfg.log_dir}"')
    print("=" * 60)



def main() -> None:
    cfg = parse_args()
    run(cfg)


if __name__ == "__main__":
    main()
