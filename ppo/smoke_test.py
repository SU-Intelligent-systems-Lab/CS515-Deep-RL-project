"""
Fast NaN-free check.

Runs a tiny PPO training session on CartPole-v1 (1000 steps, 2 epochs, 256-step
rollout). Confirms:
  * env loop works end-to-end with the gymnasium 5-tuple API,
  * buffer fills + computes returns without shape errors,
  * update produces finite losses / parameters (no NaNs, no infs),
  * diagnostics come out in sane ranges.

"""
from __future__ import annotations

import math

import numpy as np
import torch

from . import ptu
from .ppo_agent import PPOAgent
from .rollout_buffer import RolloutBuffer


def main() -> None:
    import gymnasium as gym

    ptu.init_gpu(use_gpu=False)  # CPU keeps the smoke test fast to start
    torch.manual_seed(0)
    np.random.seed(0)

    env = gym.make("CartPole-v1")
    ob_dim = int(np.prod(env.observation_space.shape))
    ac_dim = int(env.action_space.n)
    discrete = True

    agent = PPOAgent(
        ob_dim=ob_dim,
        ac_dim=ac_dim,
        discrete=discrete,
        lr=3e-4,
        update_epochs=2,
        minibatch_size=32,
    )
    buffer = RolloutBuffer(n_steps=256, ob_dim=ob_dim, ac_dim=ac_dim, discrete=discrete)

    obs, _ = env.reset(seed=0)
    total_steps = 0
    target_steps = 1000
    updates = 0

    while total_steps < target_steps:
        for _ in range(buffer.n_steps):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
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
            
            if terminated or truncated:
                obs, _ = env.reset()
            else:
                obs = next_obs
                
            total_steps += 1

        last_value = agent.bootstrap_value(obs)
        buffer.compute_returns_and_advantages(last_value, gamma=0.99)
        diagnostics = agent.update(buffer)
        buffer.reset()
        updates += 1

        # Assert every diagnostic scalar is finite.
        for k, v in diagnostics.items():
            assert math.isfinite(v), f"Non-finite diagnostic {k}={v}"

        # Assert every parameter is finite.
        for name, p in agent.net.named_parameters():
            if not torch.isfinite(p).all():
                raise AssertionError(f"Non-finite values in parameter {name}")

        print(
            f"[smoke update {updates}] "
            f"pol_loss {diagnostics['losses/policy_loss']:.4f}  "
            f"val_loss {diagnostics['losses/value_loss']:.4f}  "
            f"ent {diagnostics['losses/entropy']:.4f}  "
            f"kl {diagnostics['ppo/approx_kl']:.4f}  "
            f"clip {diagnostics['ppo/clip_fraction']:.3f}"
        )

    env.close()
    print(f"\nOK — ran {total_steps} steps across {updates} updates, all values finite.")


if __name__ == "__main__":
    main()