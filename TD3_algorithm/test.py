"""
Load a trained TD3 model and watch it play (or record it as video).

Usage:
    # Watch at real-time speed (default)
    python test.py --env_name Ant-v4 --load_dir 
    "F:\master\SabanciCourse\Spring_2025_2026\DeepLearning\Project\TD3_algorithm\runs\td3_Ant-v4_28-04-2026_04-41-52"
         --record_video --episodes 3 --video_dir "F:\master\SabanciCourse\Spring_2025_2026\DeepLearning\Project\TD3_algorithm\test_videos"
         --max_episode_steps 300

    # Cap each episode at 200 steps (useful for short demo clips)
    python test.py --env_name Ant-v4 --load_dir ... --max_episode_steps 200

    # Record videos instead of watching live
    python test.py --env_name Ant-v4 --load_dir ... --record_video --episodes 3

    # Both: cap each at 300 steps and save to mp4
    python test.py --env_name Ant-v4 --load_dir ... \
                   --max_episode_steps 300 --record_video --episodes 5
"""

import argparse
import os
import time
import numpy as np
import torch
import gym

from agent import TD3Agent


# Frame time per environment step (seconds of "real-world" simulated time
# per env.step() call), used for real-time playback pacing.
DEFAULT_DT = {
    # MuJoCo (v4/v5)
    'Ant':                       0.05,
    'HalfCheetah':               0.05,
    'Hopper':                    0.008,
    'Walker2d':                  0.008,
    'Reacher':                   0.02,
    'InvertedPendulum':          0.04,
    'InvertedDoublePendulum':    0.05,
    'Humanoid':                  0.015,
    'Swimmer':                   0.04,
    # Box2D
    'LunarLanderContinuous':     1.0 / 50,
    'BipedalWalker':             1.0 / 50,
    'CarRacing':                 1.0 / 50,
    # Classic control
    'Pendulum':                  0.05,
}


def _make_env(env_name, render=True, record_video=False,
              video_dir=None, video_prefix='test', max_episode_steps=0):
    """
    Build the test env, optionally with:
      - on-screen rendering
      - video recording (saves mp4 to video_dir)
      - a custom per-episode step limit
    """
    if record_video:
        render_mode = "rgb_array"   # frames captured by RecordVideo
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    try:
        env = gym.make(env_name, render_mode=render_mode)
    except TypeError:
        env = gym.make(env_name)

    # Custom time limit
    if max_episode_steps and max_episode_steps > 0:
        try:
            from gym.wrappers import TimeLimit
        except ImportError:
            try:
                from gymnasium.wrappers import TimeLimit
            except ImportError:
                TimeLimit = None
        if TimeLimit is not None:
            # Strip any existing TimeLimit wrapper, then add our own
            inner = env
            while (hasattr(inner, 'env') and
                   inner.__class__.__name__ == 'TimeLimit'):
                inner = inner.env
            env = TimeLimit(inner, max_episode_steps=max_episode_steps)

    # Video recording
    if record_video:
        try:
            from gym.wrappers import RecordVideo
        except ImportError:
            try:
                from gymnasium.wrappers import RecordVideo
            except ImportError:
                RecordVideo = None
        if RecordVideo is None:
            print("[video] WARNING: RecordVideo wrapper not available - "
                  "video recording disabled.")
        else:
            os.makedirs(video_dir, exist_ok=True)
            # episode_trigger=lambda x: True -> record EVERY episode
            env = RecordVideo(
                env,
                video_folder=video_dir,
                episode_trigger=lambda ep_id: True,
                name_prefix=video_prefix,
            )

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


def _get_step_dt(env, env_name):
    """Best-effort lookup of seconds per env.step()."""
    try:
        unwrapped = env.unwrapped
        if hasattr(unwrapped, 'model') and hasattr(unwrapped, 'frame_skip'):
            return float(unwrapped.model.opt.timestep) * \
                   int(unwrapped.frame_skip)
        if hasattr(unwrapped, 'dt'):
            return float(unwrapped.dt)
    except Exception:
        pass
    base = env_name.split('-')[0]
    if base in DEFAULT_DT:
        return DEFAULT_DT[base]
    return 1.0 / 50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, required=True)
    parser.add_argument('--load_dir', type=str, required=True)
    parser.add_argument('--model_prefix', type=str, default='model')
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--no_render', action='store_true',
                        help='Disable on-screen rendering '
                             '(automatic when --record_video is set)')

    # NEW: testing-side time-limit and video recording
    parser.add_argument('--max_episode_steps', type=int, default=0,
                        help='Cap each test episode at N steps '
                             '(0 = use the env\'s own default time limit)')
    parser.add_argument('--record_video', action='store_true',
                        help='Save mp4 videos of each test episode to '
                             '<load_dir>/test_videos/. '
                             'Requires moviepy or imageio[ffmpeg].')
    parser.add_argument('--video_dir', type=str, default='',
                        help='Where to save videos (default: '
                             '<load_dir>/test_videos)')
    parser.add_argument('--video_prefix', type=str, default='test',
                        help='Filename prefix for recorded videos')

    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier for live rendering '
                             '(ignored when --record_video is set). '
                             '1.0 = real time. 0 = no throttling.')
    parser.add_argument('--fps_cap', type=float, default=None)

    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--no_gpu', action='store_true')
    args = parser.parse_args()

    # Resolve video output directory
    if args.record_video:
        if args.video_dir:
            video_dir = args.video_dir
        else:
            video_dir = os.path.join(args.load_dir, 'test_videos')
    else:
        video_dir = None

    # Build env
    env = _make_env(
        args.env_name,
        render=(not args.no_render) and (not args.record_video),
        record_video=args.record_video,
        video_dir=video_dir,
        video_prefix=args.video_prefix,
        max_episode_steps=args.max_episode_steps,
    )

    ob_dim     = env.observation_space.shape[0]
    ac_dim     = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    # Pacing (only matters for live rendering, not for video recording)
    step_dt = _get_step_dt(env, args.env_name)
    if args.record_video or args.no_render or args.speed <= 0:
        target_dt = 0.0   # don't throttle
    else:
        target_dt = step_dt / args.speed
        if args.fps_cap is not None and args.fps_cap > 0:
            target_dt = max(target_dt, 1.0 / args.fps_cap)

    # Print config
    print("--------------------------------------------------")
    print(f"Env:               {args.env_name}")
    print(f"Env step dt:       {step_dt*1000:.2f} ms "
          f"({1.0/step_dt:.1f} env Hz)")
    if args.max_episode_steps and args.max_episode_steps > 0:
        print(f"Episode limit:     {args.max_episode_steps} steps")
    if args.record_video:
        print(f"Recording videos:  ON  -> {video_dir}")
    elif target_dt > 0:
        print(f"Playback target:   {1.0/target_dt:.1f} Hz "
              f"(speed = {args.speed}x)")
    else:
        print("Playback:          raw simulator speed")
    print("--------------------------------------------------")

    # Build agent and load
    agent = TD3Agent(env, {
        'ob_dim':        ob_dim,
        'ac_dim':        ac_dim,
        'max_action':    max_action,
        'gamma':         0.99,
        'tau':           0.005,
        'policy_noise':  0.2 * max_action,
        'noise_clip':    0.5 * max_action,
        'policy_freq':   2,
        'batch_size':    256,
        'n_layers':      args.n_layers,
        'size':          args.size,
        'learning_rate': 3e-4,
        'buffer_size':   1000,
        'no_gpu':        args.no_gpu,
    })

    model_path = os.path.join(args.load_dir, args.model_prefix)
    agent.load(model_path)
    print(f"Loaded model from {model_path}")

    # Play
    total_rewards = []
    for ep in range(args.episodes):
        state = _reset_env(env)
        ep_reward = 0.0
        steps     = 0
        done      = False

        while not done:
            loop_start = time.perf_counter()

            action = agent.select_action(np.array(state), noise=0.0)
            state, reward, done, _ = _step_env(env, action)

            # Live render (only when not recording)
            if not args.record_video and not args.no_render:
                _try_render(env)

            ep_reward += reward
            steps     += 1

            if target_dt > 0:
                elapsed   = time.perf_counter() - loop_start
                sleep_for = target_dt - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{args.episodes} | steps: {steps:>4} "
              f"| reward: {ep_reward:>8.2f}")

    print("-" * 50)
    print(f"Average over {args.episodes} episodes: "
          f"{np.mean(total_rewards):.2f} ± {np.std(total_rewards):.2f}")
    if args.record_video:
        print(f"Videos saved to: {video_dir}")
    env.close()


if __name__ == '__main__':
    main()