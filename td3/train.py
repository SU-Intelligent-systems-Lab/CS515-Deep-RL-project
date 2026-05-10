"""
TD3 trainer
"""

import os
import csv
import json
import numpy as np
import torch
import gymnasium as gym
import dataclasses


from td3.agent import TD3Agent
from td3.utils import eval_policy
from td3.config import TD3Config

# Gym API compatibility function
def _make_env(env_name, render=False, seed=None):
    try:
        env = gym.make(env_name, render_mode="human" if render else None)
    except TypeError:
        env = gym.make(env_name)
    if seed is not None:
        env.action_space.seed(seed)
    return env



def _try_render(env):
    try:
        env.render()
    except Exception:
        pass

# Plotting
def plot_results(episode_rewards, eval_rewards, eval_timesteps, logdir,
                 show=True):
    """
    Plot the training curves for episode rewards and evaluation performance.

    Backend selection:
      - If `show=True`, use the default interactive backend.
      - If `show=False` or no display is detected, use 'Agg' (file only).
    """
    try:
        import matplotlib
    except ImportError:
        print("[plot] matplotlib not installed; skipping plots.")
        return

    # Pick a backend BEFORE importing pyplot
    use_interactive = show and _has_display()
    if not use_interactive:
        matplotlib.use('Agg')

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Per-episode training reward 
    ax = axes[0]
    if len(episode_rewards) > 0:
        ep = np.arange(1, len(episode_rewards) + 1)
        ax.plot(ep, episode_rewards, alpha=0.4, label='per episode')
        if len(episode_rewards) >= 10:
            window = min(50, max(1, len(episode_rewards) // 5))
            ma = np.convolve(episode_rewards,
                             np.ones(window) / window, mode='valid')
            ax.plot(ep[window - 1:], ma, linewidth=2,
                    label=f'{window}-episode moving avg')
        ax.legend()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total reward')
    ax.set_title('Training reward per episode')
    ax.grid(True, alpha=0.3)

    # Evaluation returns 
    ax = axes[1]
    if len(eval_rewards) > 0:
        ax.plot(eval_timesteps, eval_rewards, marker='o', linewidth=2)
    ax.set_xlabel('Environment steps')
    ax.set_ylabel('Average return (eval)')
    ax.set_title('Evaluation performance')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(logdir, 'training_curves.png')
    plt.savefig(out_path, dpi=120)
    print(f"[plot] Saved plot to {out_path}")

    # if use_interactive:
    #     plt.show()
    # plt.close(fig)


def _has_display():
    """True if a GUI display is likely available."""
    if os.name == 'nt':                          # Windows: assume yes
        return True
    if os.environ.get('DISPLAY'):                # Linux with X
        return True
    if os.environ.get('WAYLAND_DISPLAY'):        # Linux with Wayland
        return True
    return False


# History loggingd
def write_episode_csv(path, episode_rewards, episode_lengths,
                      episode_timesteps_at_end):
    """
    Write a CSV with per-episode training history:
        episode, steps, total_steps, reward
    """
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'episode_length',
                    'total_env_steps', 'total_reward'])
        for i, (r, ln, t) in enumerate(zip(
                episode_rewards, episode_lengths,
                episode_timesteps_at_end)):
            w.writerow([i + 1, ln, t, f"{r:.4f}"])


def write_eval_csv(path, eval_rewards, eval_timesteps):
    """
    Write a CSV with evaluation history:
        env_step, avg_eval_reward
    """
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['env_step', 'avg_eval_reward'])
        for t, r in zip(eval_timesteps, eval_rewards):
            w.writerow([t, f"{r:.4f}"])


def write_summary_json(path, cfg: TD3Config, episode_rewards, eval_rewards,
                       eval_timesteps, total_steps_done):
    """
    One-shot JSON summary of the whole run.
    """
    cfg_dict = dataclasses.asdict(cfg)
    summary = {
        'env_name':           cfg.env_name,
        'exp_name':           cfg.exp_name,
        'seed':               cfg.seed,
        'max_timesteps':      cfg.max_timesteps,
        'total_steps_done':   int(total_steps_done),
        'num_episodes':       len(episode_rewards),
        'num_evaluations':    len(eval_rewards),
        'episode_reward_mean':  float(np.mean(episode_rewards))
                                if episode_rewards else None,
        'episode_reward_last': float(episode_rewards[-1])
                                if episode_rewards else None,
        'episode_reward_best': float(np.max(episode_rewards))
                                if episode_rewards else None,
        'eval_reward_last':    float(eval_rewards[-1])
                                if eval_rewards else None,
        'eval_reward_best':    float(np.max(eval_rewards))
                                if eval_rewards else None,
        'hyperparameters':    {k: v for k, v in cfg_dict.items()
                               if isinstance(v, (int, float, str, bool))
                               and k != 'logdir'},
    }
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)


def run(cfg: TD3Config) ->None:
    # Build env
    env = _make_env(cfg.env_name,
                    render=cfg.render,
                    seed=cfg.seed)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    assert hasattr(env.action_space, 'high'), \
        "TD3 only supports continuous action spaces."

    ob_dim     = env.observation_space.shape[0]
    ac_dim     = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print("--------------------------------------------------")
    print(f"Env:        {cfg.env_name}")
    print(f"  ob_dim    = {ob_dim}")
    print(f"  ac_dim    = {ac_dim}")
    print(f"  max_action= {max_action}")
    print(f"Logdir:     {cfg.log_dir}")
    if cfg.render:
        print(f"Rendering:  every {cfg.render_freq} episode(s)")
    else:
        print("Rendering:  off")
    print("--------------------------------------------------")


    # Build agent
    agent_params = {
        'ob_dim':        ob_dim,
        'ac_dim':        ac_dim,
        'max_action':    max_action,
        'gamma':         cfg.discount,
        'tau':           cfg.tau,
        'policy_noise':  cfg.policy_noise * max_action,
        'noise_clip':    cfg.noise_clip   * max_action,
        'policy_freq':   cfg.policy_freq,
        'batch_size':    cfg.batch_size,
        'n_layers':      cfg.n_layers,
        'size':          cfg.size,
        'learning_rate': cfg.learning_rate,
        'buffer_size':   cfg.buffer_size,
        'no_gpu':        cfg.no_gpu,
    }
    agent = TD3Agent(env, agent_params)

    if cfg.load_model:
        agent.load(cfg.load_model)
        print(f"Loaded model from {cfg.load_model}")

    # Tracking
    episode_rewards            = []  # total reward per training episode
    episode_lengths            = []  # number of steps in each episode
    episode_timesteps_at_end   = []  # cumulative env step at episode end
    eval_rewards               = []
    eval_timesteps             = []

    # File paths 
    csv_train_path   = os.path.join(cfg.log_dir, 'episode_rewards.csv')
    csv_eval_path    = os.path.join(cfg.log_dir, 'eval_rewards.csv')
    summary_path     = os.path.join(cfg.log_dir, 'summary.json')

    # Initial eval
    eval_rewards.append(eval_policy(agent, cfg.env_name,
                                    cfg.seed,
                                    eval_episodes=cfg.eval_episodes))
    eval_timesteps.append(0)

    # Training loop
    state, _ = env.reset(seed=cfg.seed)
    episode_reward    = 0.0
    episode_timesteps = 0
    episode_num       = 0
    render_this_ep    = cfg.render and (episode_num % cfg.render_freq == 0)
    max_ep_steps      = getattr(env, '_max_episode_steps', None)
    total_steps_done  = 0

    try:
        for t in range(int(cfg.max_timesteps)):
            episode_timesteps += 1
            total_steps_done   = t + 1

            if t < cfg.start_timesteps:
                action = env.action_space.sample()
            else:
                action = agent.select_action(np.array(state),
                                             noise=cfg.expl_noise)

            next_state, reward, terminated, truncated, info = env.step(action)

            if render_this_ep:
                _try_render(env)

            done_bool = float(terminated)

            agent.add_to_replay_buffer(state, action, next_state,
                                       reward, done_bool)
            state = next_state
            episode_reward += reward

            if t >= cfg.start_timesteps:
                agent.train()

            # End of episode
            if terminated or truncated:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_timesteps)
                episode_timesteps_at_end.append(t + 1)

                # Append to CSV (rewrite full file = simple + crash-safe)
                write_episode_csv(csv_train_path, episode_rewards,
                                  episode_lengths, episode_timesteps_at_end)

                print(f"Total T: {t+1:>7} | Ep: {episode_num+1:>4} "
                      f"| Steps: {episode_timesteps:>4} "
                      f"| Reward: {episode_reward:>8.2f}")

                state, _ = env.reset()
                episode_reward    = 0.0
                episode_timesteps = 0
                episode_num      += 1
                render_this_ep    = (cfg.render and
                                     episode_num % cfg.render_freq == 0)

            # Periodic evaluation
            if (t + 1) % cfg.eval_freq == 0:
                avg = eval_policy(agent, cfg.env_name,
                                  cfg.seed,
                                  eval_episodes=cfg.eval_episodes)
                eval_rewards.append(avg)
                eval_timesteps.append(t + 1)
                write_eval_csv(csv_eval_path, eval_rewards, eval_timesteps)

    except KeyboardInterrupt:
        print("\n[main] Training interrupted - saving current state...")


    # Final saves
    if cfg.save_model:
        model_path = os.path.join(cfg.log_dir, 'model')
        agent.save(model_path)
        print(f"\n[main] Model saved:")
        print(f"        actor  -> {model_path}_actor")
        print(f"        critic -> {model_path}_critic")

    # Final flush of training history CSVs
    write_episode_csv(csv_train_path, episode_rewards,
                      episode_lengths, episode_timesteps_at_end)
    write_eval_csv(csv_eval_path, eval_rewards, eval_timesteps)
    print(f"[main] Training history -> {csv_train_path}")
    print(f"[main] Eval history     -> {csv_eval_path}")

    # JSON summary
    write_summary_json(summary_path, cfg, episode_rewards,
                       eval_rewards, eval_timesteps, total_steps_done)
    print(f"[main] Summary          -> {summary_path}")

    # Keep .npy version
    np.save(os.path.join(cfg.log_dir, 'episode_rewards.npy'),
            np.array(episode_rewards))
    np.save(os.path.join(cfg.log_dir, 'eval_rewards.npy'),
            np.array(eval_rewards))
    np.save(os.path.join(cfg.log_dir, 'eval_timesteps.npy'),
            np.array(eval_timesteps))

    # Plot
    if cfg.plot:
        plot_results(episode_rewards, eval_rewards, eval_timesteps,
                     cfg.log_dir, show=cfg.show_plot)

    env.close()

    print("\n" + "=" * 60)
    print("Training complete. To watch the trained agent play:")
    print(f'  python main.py --mode test --load_dir {cfg.env_name} '
          f'--load_dir "{cfg.log_dir}"')
    print("=" * 60)

