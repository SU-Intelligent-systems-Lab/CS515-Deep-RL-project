"""
Main training loop for TD3 with live rendering, model saving,
training history logging (CSV + JSON)

Examples
--------
    python main.py --env_name LunarLanderContinuous-v2 --render
    python main.py --env_name LunarLanderContinuous-v2 --render --render_freq 10
    python main.py --env_name Pendulum-v1 --start_timesteps 1000 --max_timesteps 100000
"""

import os
import csv
import json
import numpy as np
import torch
import gym

from agent import TD3Agent
from parameters import arg_parse
from utils import eval_policy


# Gym API compatibility function
def _make_env(env_name, render=False, seed=None):
    try:
        env = gym.make(env_name, render_mode="human" if render else None)
    except TypeError:
        env = gym.make(env_name)
    if seed is not None:
        try:
            env.seed(seed)
            env.action_space.seed(seed)
        except AttributeError:
            pass
    return env


def _reset_env(env):
    out = env.reset()
    return out[0] if isinstance(out, tuple) else out


def _step_env(env, action):
    out = env.step(action)
    if len(out) == 5:
        obs, r, terminated, truncated, info = out
        return obs, r, (terminated or truncated), info
    return out


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


# History logging
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


def write_summary_json(path, params, episode_rewards, eval_rewards,
                       eval_timesteps, total_steps_done):
    """
    One-shot JSON summary of the whole run.
    """
    summary = {
        'env_name':           params['env_name'],
        'exp_name':           params['exp_name'],
        'seed':               params['seed'],
        'max_timesteps':      params['max_timesteps'],
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
        'hyperparameters':    {k: v for k, v in params.items()
                               if isinstance(v, (int, float, str, bool))
                               and k != 'logdir'},
    }
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)


def main():
    params = arg_parse()

    # Save raw hyperparameters
    with open(os.path.join(params['logdir'], 'params.json'), 'w') as f:
        json.dump({k: v for k, v in params.items()
                   if isinstance(v, (int, float, str, bool, list))},
                  f, indent=2)
    # Build env
    env = _make_env(params['env_name'],
                    render=params['render'],
                    seed=params['seed'])

    torch.manual_seed(params['seed'])
    np.random.seed(params['seed'])

    assert hasattr(env.action_space, 'high'), \
        "TD3 only supports continuous action spaces."

    ob_dim     = env.observation_space.shape[0]
    ac_dim     = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    print("--------------------------------------------------")
    print(f"Env:        {params['env_name']}")
    print(f"  ob_dim    = {ob_dim}")
    print(f"  ac_dim    = {ac_dim}")
    print(f"  max_action= {max_action}")
    print(f"Logdir:     {params['logdir']}")
    if params['render']:
        print(f"Rendering:  every {params['render_freq']} episode(s)")
    else:
        print("Rendering:  off")
    print("--------------------------------------------------")


    # Build agent
    agent_params = {
        'ob_dim':        ob_dim,
        'ac_dim':        ac_dim,
        'max_action':    max_action,
        'gamma':         params['discount'],
        'tau':           params['tau'],
        'policy_noise':  params['policy_noise'] * max_action,
        'noise_clip':    params['noise_clip']   * max_action,
        'policy_freq':   params['policy_freq'],
        'batch_size':    params['batch_size'],
        'n_layers':      params['n_layers'],
        'size':          params['size'],
        'learning_rate': params['learning_rate'],
        'buffer_size':   params['buffer_size'],
        'no_gpu':        params['no_gpu'],
    }
    agent = TD3Agent(env, agent_params)

    if params['load_model']:
        agent.load(params['load_model'])
        print(f"Loaded model from {params['load_model']}")

    # Tracking
    episode_rewards            = []  # total reward per training episode
    episode_lengths            = []  # number of steps in each episode
    episode_timesteps_at_end   = []  # cumulative env step at episode end
    eval_rewards               = []
    eval_timesteps             = []

    # File paths 
    csv_train_path   = os.path.join(params['logdir'], 'episode_rewards.csv')
    csv_eval_path    = os.path.join(params['logdir'], 'eval_rewards.csv')
    summary_path     = os.path.join(params['logdir'], 'summary.json')

    # Initial eval
    eval_rewards.append(eval_policy(agent, params['env_name'],
                                    params['seed'],
                                    eval_episodes=params['eval_episodes']))
    eval_timesteps.append(0)

    # Training loop
    state = _reset_env(env)
    episode_reward    = 0.0
    episode_timesteps = 0
    episode_num       = 0
    render_this_ep    = params['render'] and (episode_num % params['render_freq'] == 0)
    max_ep_steps      = getattr(env, '_max_episode_steps', None)
    total_steps_done  = 0

    try:
        for t in range(int(params['max_timesteps'])):
            episode_timesteps += 1
            total_steps_done   = t + 1

            if t < params['start_timesteps']:
                action = env.action_space.sample()
            else:
                action = agent.select_action(np.array(state),
                                             noise=params['expl_noise'])

            next_state, reward, done, _ = _step_env(env, action)

            if render_this_ep:
                _try_render(env)

            if max_ep_steps is not None and episode_timesteps >= max_ep_steps:
                done_bool = 0.0
            else:
                done_bool = float(done)

            agent.add_to_replay_buffer(state, action, next_state,
                                       reward, done_bool)
            state = next_state
            episode_reward += reward

            if t >= params['start_timesteps']:
                agent.train()

            # End of episode
            if done:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_timesteps)
                episode_timesteps_at_end.append(t + 1)

                # Append to CSV (rewrite full file = simple + crash-safe)
                write_episode_csv(csv_train_path, episode_rewards,
                                  episode_lengths, episode_timesteps_at_end)

                print(f"Total T: {t+1:>7} | Ep: {episode_num+1:>4} "
                      f"| Steps: {episode_timesteps:>4} "
                      f"| Reward: {episode_reward:>8.2f}")

                state = _reset_env(env)
                episode_reward    = 0.0
                episode_timesteps = 0
                episode_num      += 1
                render_this_ep    = (params['render'] and
                                     episode_num % params['render_freq'] == 0)

            # Periodic evaluation
            if (t + 1) % params['eval_freq'] == 0:
                avg = eval_policy(agent, params['env_name'],
                                  params['seed'],
                                  eval_episodes=params['eval_episodes'])
                eval_rewards.append(avg)
                eval_timesteps.append(t + 1)
                write_eval_csv(csv_eval_path, eval_rewards, eval_timesteps)

    except KeyboardInterrupt:
        print("\n[main] Training interrupted - saving current state...")


    # Final saves
    if params['save_model']:
        model_path = os.path.join(params['logdir'], 'model')
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
    write_summary_json(summary_path, params, episode_rewards,
                       eval_rewards, eval_timesteps, total_steps_done)
    print(f"[main] Summary          -> {summary_path}")

    # Keep .npy version
    np.save(os.path.join(params['logdir'], 'episode_rewards.npy'),
            np.array(episode_rewards))
    np.save(os.path.join(params['logdir'], 'eval_rewards.npy'),
            np.array(eval_rewards))
    np.save(os.path.join(params['logdir'], 'eval_timesteps.npy'),
            np.array(eval_timesteps))

    # Plot
    if params['plot']:
        plot_results(episode_rewards, eval_rewards, eval_timesteps,
                     params['logdir'], show=params['show_plot'])

    env.close()

    print("\n" + "=" * 60)
    print("Training complete. To watch the trained agent play:")
    print(f'  python test.py --env_name {params["env_name"]} '
          f'--load_dir "{params["logdir"]}"')
    print("=" * 60)


if __name__ == '__main__':
    main()