"""
Utilities for TD3: replay buffer and policy evaluation.
"""

import numpy as np
import torch
import gymnasium as gym


def get_device(no_gpu=False):
    if no_gpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


class ReplayBuffer(object):
    """
    Standard off-policy transition buffer for TD3.

    TD3 samples individual (s, a, r, s', done) transitions uniformly at random.
    """

    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0
        self.device = device if device is not None else torch.device("cpu")

        self.state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((self.max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((self.max_size, state_dim), dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, terminated):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - float(terminated)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.as_tensor(self.state[ind], device=self.device),
            torch.as_tensor(self.action[ind], device=self.device),
            torch.as_tensor(self.next_state[ind], device=self.device),
            torch.as_tensor(self.reward[ind], device=self.device),
            torch.as_tensor(self.not_done[ind], device=self.device),
        )

    def __len__(self):
        return self.size


def eval_policy(agent, env_name, seed, eval_episodes=10):
    """
    Evaluate the current deterministic policy (no exploration noise)
    """
    eval_env = gym.make(env_name)

    avg_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = eval_env.reset(seed=seed + 100)
        done = False
        while not done:
            action = agent.select_action(np.array(state), noise=0.0)
            state, reward, terminated, truncated, _ = eval_env.step(action)                 
            done = terminated or truncated
            avg_reward += reward

    avg_reward /= eval_episodes
    eval_env.close()

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward