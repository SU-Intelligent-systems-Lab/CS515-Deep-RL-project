"""
Double-DQN training loop
"""
import csv
import json
import os
from collections import deque
from typing import List, Optional

import gymnasium as gym
import matplotlib
import numpy as np
import torch

from DDQN.config     import DDQNConfig
from DDQN.ddqn_agent import DQNAgent


def _has_display():
    if os.name == "nt":
        return True
    if os.environ.get("DISPLAY"):
        return True
    if os.environ.get("WAYLAND_DISPLAY"):
        return True
    return False


# Plotting                                                                    
def plot_scores(scores: List[float], title: str = "Training Progress",
                save_path: Optional[str] = None, show: bool = True) -> None:
    """Plots the training scores. Saves to disk and/or displays interactively."""
    use_interactive = show and _has_display()
    if not use_interactive:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(scores) + 1), scores, alpha=0.4,
            label="per episode")
    if len(scores) >= 10:
        window = min(100, max(1, len(scores) // 5))
        ma = np.convolve(scores, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(window, len(scores) + 1), ma, linewidth=2,
                label=f"{window}-episode moving avg")
        ax.legend()

    ax.set_xlabel("Episode")
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=120)
        print(f"[plot] Saved plot to {save_path}")
    if use_interactive:
        plt.show()
    plt.close(fig)



# CSV / JSON writers                                                          
def _write_scores_csv(path: str, scores: List[float]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "score"])
        for i, s in enumerate(scores):
            w.writerow([i + 1, f"{s:.4f}"])


def _write_summary_json(path: str, cfg: DDQNConfig, scores: List[float],
                        solved_at: Optional[int]) -> None:
    summary = {
        "env_name":          cfg.env_name,
        "exp_name":          cfg.exp_name,
        "seed":              cfg.seed,
        "use_ddqn":          cfg.use_ddqn,
        "n_episodes_run":    len(scores),
        "n_episodes_target": cfg.n_episodes,
        "score_threshold":   cfg.score_threshold,
        "solved_at_episode": solved_at,
        "score_mean":        float(np.mean(scores)) if scores else None,
        "score_last":        float(scores[-1]) if scores else None,
        "score_best":        float(np.max(scores)) if scores else None,
        "score_last_100_mean": (float(np.mean(scores[-100:]))
                                if len(scores) >= 1 else None),
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)


# Core training loop                                                          
def train_agent(env_name: str = "CartPole-v1", n_episodes: int = 1000,
                max_t: int = 1000, eps_start: float = 1.0,
                eps_end: float = 0.01, eps_decay: float = 0.995,
                use_ddqn: bool = True, seed: int = 0,
                score_threshold: Optional[float] = 475.0,
                save_path: Optional[str] = None):
    """Deep Q-Learning training loop.

    Returns (scores, solved_at_episode_or_None).
    """
    print(f"Starting Training on {env_name} | DDQN Enabled: {use_ddqn}")

    env = gym.make(env_name)
    state_size  = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size=state_size, action_size=action_size,
                     seed=seed, use_ddqn=use_ddqn)

    scores: List[float] = []
    scores_window: deque = deque(maxlen=100)
    eps = eps_start
    solved_at: Optional[int] = None

    for i_episode in range(1, n_episodes + 1):
        state, _info = env.reset()
        score = 0.0

        for _t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state  = next_state
            score += reward
            if done:
                break

        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)

        print(f"\rEpisode {i_episode}\tAverage Score: "
              f"{np.mean(scores_window):.2f}\tEpsilon: {eps:.2f}", end="")
        if i_episode % 100 == 0:
            print(f"\rEpisode {i_episode}\tAverage Score: "
                  f"{np.mean(scores_window):.2f}")

        if (score_threshold is not None
                and len(scores_window) == 100
                and np.mean(scores_window) >= score_threshold):
            solved_at = max(1, i_episode - 100)
            print(f"\nEnvironment solved in {solved_at} episodes!\t"
                  f"Average Score: {np.mean(scores_window):.2f}")
            break
    if save_path is None:
        save_path = f"checkpoint_{'ddqn' if use_ddqn else 'dqn'}.pth"
    torch.save(agent.qnetwork_local.state_dict(), save_path)
    print(f"\n[train] Q-network saved -> {save_path}")

    env.close()
    return scores, solved_at



# Entry point called by root main.py                                          

def run(cfg: DDQNConfig) -> None:
    """Train Double DQN with hyperparameters from the unified DDQNConfig."""
    os.makedirs(cfg.log_dir, exist_ok=True)
    model_path   = os.path.join(cfg.log_dir, "model.pth")
    scores_path  = os.path.join(cfg.log_dir, "episode_scores.csv")
    summary_path = os.path.join(cfg.log_dir, "summary.json")
    plot_path    = os.path.join(cfg.log_dir, "training_curves.png")

    scores, solved_at = train_agent(
        env_name        = cfg.env_name,
        n_episodes      = cfg.n_episodes,
        max_t           = cfg.max_t,
        eps_start       = cfg.eps_start,
        eps_end         = cfg.eps_end,
        eps_decay       = cfg.eps_decay,
        use_ddqn        = cfg.use_ddqn,
        seed            = cfg.seed,
        score_threshold = cfg.score_threshold,
        save_path       = model_path,
    )

    _write_scores_csv(scores_path, scores)
    print(f"[train] Episode scores  -> {scores_path}")

    _write_summary_json(summary_path, cfg, scores, solved_at)
    print(f"[train] Summary         -> {summary_path}")

    np.save(os.path.join(cfg.log_dir, "episode_scores.npy"),
            np.array(scores, dtype=np.float32))

    if cfg.plot:
        try:
            algo = "DDQN" if cfg.use_ddqn else "DQN"
            # plot_scores(scores, title=f"{algo} on {cfg.env_name}",
                        # save_path=plot_path, show=cfg.show_plot)
        except Exception as e:
            print(f"[plot] failed: {e}")

    print("\n" + "=" * 60)
    print("Training complete. To watch the trained agent play:")
    print(f'  python main.py --mode test --load_dir "{cfg.log_dir}"')
    print("=" * 60)


if __name__ == "__main__":
    scores, _ = train_agent(n_episodes=1000, use_ddqn=True)
    # plot_scores(scores, title="DDQN on CartPole-v1")