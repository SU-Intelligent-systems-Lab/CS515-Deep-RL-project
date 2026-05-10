"""
Unified test/play script for any trained DRL agent.

Examples
--------
    # Live render, 5 episodes
    python test.py --load_dir runs/td3/td3_Pendulum-v1_28-04-2026_04-41-52

    # Record videos, 3 episodes, capped at 300 steps each
    python test.py --load_dir runs/td3/td3_Ant-v4_... \
                   --record_video --episodes 3 --max_episode_steps 300

    # PPO test
    python test.py --load_dir runs/ppo/ppo_CartPole-v1_...

    # Override auto-detected algo
    python test.py --load_dir <run> --algo ppo
"""

import argparse
import json
import os
import time

import numpy as np

# Default per-env step times (used only for live-rendering pacing)            
DEFAULT_DT = {
    "Ant": 0.05, "HalfCheetah": 0.05, "Hopper": 0.008, "Walker2d": 0.008,
    "Reacher": 0.02, "InvertedPendulum": 0.04,
    "InvertedDoublePendulum": 0.05, "Humanoid": 0.015, "Swimmer": 0.04,
    "LunarLanderContinuous": 1/50, "LunarLander": 1/50,
    "BipedalWalker": 1/50, "CarRacing": 1/50,
    "Pendulum": 0.05, "CartPole": 0.02,
    "MountainCar": 0.02, "MountainCarContinuous": 0.02, "Acrobot": 0.2,
}

# gym/gymnasium helpers                                                       
def _import_gym(prefer_gymnasium=False):
    """Return the gym module (gymnasium first if requested, gym otherwise)."""
    if prefer_gymnasium:
        try:
            import gymnasium as gym
            return gym
        except ImportError:
            pass
    try:
        import gym
        return gym
    except ImportError:
        import gymnasium as gym
        return gym


def _make_env(env_name, render=True, record_video=False,
              video_dir=None, video_prefix="test", max_episode_steps=0,
              prefer_gymnasium=False):
    gym = _import_gym(prefer_gymnasium=prefer_gymnasium)

    if record_video:
        render_mode = "rgb_array"
    elif render:
        render_mode = "human"
    else:
        render_mode = None

    try:
        env = gym.make(env_name, render_mode=render_mode)
    except TypeError:
        env = gym.make(env_name)

    if max_episode_steps and max_episode_steps > 0:
        TimeLimit = None
        try:
            from gym.wrappers import TimeLimit
        except ImportError:
            try:
                from gymnasium.wrappers import TimeLimit
            except ImportError:
                TimeLimit = None
        if TimeLimit is not None:
            inner = env
            while (hasattr(inner, "env") and
                   inner.__class__.__name__ == "TimeLimit"):
                inner = inner.env
            env = TimeLimit(inner, max_episode_steps=max_episode_steps)

    if record_video:
        RecordVideo = None
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
            env = RecordVideo(
                env, video_folder=video_dir,
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
        obs, r, term, trunc, info = out
        return obs, r, (term or trunc), info
    return out


def _try_render(env):
    try:
        env.render()
    except Exception:
        pass


def _get_step_dt(env, env_name):
    try:
        u = env.unwrapped
        if hasattr(u, "model") and hasattr(u, "frame_skip"):
            return float(u.model.opt.timestep) * int(u.frame_skip)
        if hasattr(u, "dt"):
            return float(u.dt)
    except Exception:
        pass
    return DEFAULT_DT.get(env_name.split("-")[0], 1/50)


# --------------------------------------------------------------------------- #
# Per-algo agent loaders                                                      #
# Each returns (agent_or_net, action_fn) where action_fn(state) -> action.    #
# --------------------------------------------------------------------------- #
def _load_td3(env, run_params, model_dir):
    """TD3: prefix `model` -> model_actor + model_critic."""
    from td3.agent import TD3Agent

    ob_dim     = env.observation_space.shape[0]
    ac_dim     = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    agent = TD3Agent(env, {
        "ob_dim":        ob_dim,
        "ac_dim":        ac_dim,
        "max_action":    max_action,
        "gamma":         run_params.get("discount", 0.99),
        "tau":           run_params.get("tau", 0.005),
        "policy_noise":  run_params.get("policy_noise", 0.2) * max_action,
        "noise_clip":    run_params.get("noise_clip", 0.5)   * max_action,
        "policy_freq":   run_params.get("policy_freq", 2),
        "batch_size":    run_params.get("batch_size", 256),
        "n_layers":      run_params.get("n_layers", 2),
        "size":          run_params.get("size", 256),
        "learning_rate": run_params.get("learning_rate", 3e-4),
        "buffer_size":   run_params.get("buffer_size", 1000),
        "no_gpu":        run_params.get("no_gpu", False),
    })
    prefix = os.path.join(model_dir, "model")
    agent.load(prefix)
    print(f"[td3] Loaded actor+critic from prefix '{prefix}'")

    def act(state):
        return agent.select_action(np.asarray(state), noise=0.0)
    return agent, act


def _ppo_reinforce_env_spec(env):
    """Same logic as ppo/train.py:env_spec."""
    gym = _import_gym(prefer_gymnasium=True)
    ob_dim = int(np.prod(env.observation_space.shape))
    if isinstance(env.action_space, gym.spaces.Discrete):
        return ob_dim, int(env.action_space.n), True
    elif isinstance(env.action_space, gym.spaces.Box):
        return ob_dim, int(np.prod(env.action_space.shape)), False
    else:
        raise NotImplementedError(f"Unsupported action space: {env.action_space}")


def _load_ppo(env, run_params, model_dir):
    """PPO: model_final.pt is the ActorCritic state_dict."""
    import torch
    from ppo import ptu
    from ppo.networks import ActorCritic

    ptu.init_gpu(use_gpu=not run_params.get("no_gpu", False))

    ob_dim, ac_dim, discrete = _ppo_reinforce_env_spec(env)
    net = ActorCritic(
        ob_dim=ob_dim, ac_dim=ac_dim, discrete=discrete,
        n_layers=int(run_params.get("n_layers", 2)),
        size=int(run_params.get("size", 64)),
    ).to(ptu.device)

    path = os.path.join(model_dir, "model_final.pt")
    state = torch.load(path, map_location=ptu.device)
    net.load_state_dict(state)
    net.eval()
    print(f"[ppo] Loaded model from {path}")

    def act(state):
        obs = ptu.from_numpy(np.asarray(state)[None])
        with torch.no_grad():
            dist = net._distribution(obs)
            if discrete:
                a = dist.probs.argmax(dim=-1)
                return int(ptu.to_numpy(a)[0])
            else:
                a = dist.mean
                a = ptu.to_numpy(a)[0]
                return np.clip(a, env.action_space.low, env.action_space.high)
    return net, act


def _load_reinforce(env, run_params, model_dir):
    """REINFORCE: model_final.pt = {'policy': ..., 'value': ...}."""
    import torch
    from reinforce import ptu
    from reinforce.networks import PolicyNet

    ptu.init_gpu(use_gpu=not run_params.get("no_gpu", False))

    ob_dim, ac_dim, discrete = _ppo_reinforce_env_spec(env)
    policy = PolicyNet(
        ob_dim=ob_dim, ac_dim=ac_dim, discrete=discrete,
        n_layers=int(run_params.get("n_layers", 2)),
        size=int(run_params.get("size", 64)),
    ).to(ptu.device)

    path = os.path.join(model_dir, "model_final.pt")
    state = torch.load(path, map_location=ptu.device)
    policy.load_state_dict(state["policy"])
    policy.eval()
    print(f"[reinforce] Loaded policy from {path}")

    def act(state):
        obs = ptu.from_numpy(np.asarray(state)[None])
        with torch.no_grad():
            dist = policy._distribution(obs)
            if discrete:
                a = dist.probs.argmax(dim=-1)
                return int(ptu.to_numpy(a)[0])
            else:
                a = dist.mean
                a = ptu.to_numpy(a)[0]
                return np.clip(a, env.action_space.low, env.action_space.high)
    return policy, act


def _load_ddqn(env, run_params, model_dir):
    """DQN: model.pth = QNetwork state_dict; greedy argmax."""
    import torch
    from ddqn.networks import QNetwork

    state_size  = int(np.prod(env.observation_space.shape))
    action_size = int(env.action_space.n)
    seed        = int(run_params.get("seed", 0))

    qnet = QNetwork(state_size, action_size, seed)
    device = torch.device("cuda" if (torch.cuda.is_available()
                                     and not run_params.get("no_gpu", False))
                          else "cpu")
    qnet = qnet.to(device)

    path = os.path.join(model_dir, "model.pth")
    qnet.load_state_dict(torch.load(path, map_location=device))
    qnet.eval()
    print(f"[dqn] Loaded Q-network from {path}")

    def act(state):
        s = torch.from_numpy(np.asarray(state)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            q = qnet(s)
        return int(np.argmax(q.cpu().numpy()))
    return qnet, act


_LOADERS = {
    "td3":       _load_td3,
    "ppo":       _load_ppo,
    "reinforce": _load_reinforce,
    "ddqn":       _load_ddqn,
}

# Algos that train against gymnasium and may need the gymnasium reset/step API
_PREFER_GYMNASIUM = {"ppo", "reinforce", "ddqn"}


# --------------------------------------------------------------------------- #
# Run-dir discovery                                                           #
# --------------------------------------------------------------------------- #
def _detect_algo(load_dir, override=None):
    if override:
        return override.lower()
    algo_file = os.path.join(load_dir, "algo.txt")
    if os.path.isfile(algo_file):
        with open(algo_file) as f:
            return f.read().strip().lower()
    raise RuntimeError(
        f"Could not detect algorithm: no algo.txt in {load_dir!r} and "
        f"no --algo override given."
    )


def _load_run_params(load_dir):
    p = os.path.join(load_dir, "params.json")
    if os.path.isfile(p):
        with open(p) as f:
            return json.load(f)
    return {}


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def play():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_dir", required=True,
                        help="Path to the run folder created during training")
    parser.add_argument("--algo", default=None,
                        help="Override auto-detected algorithm")
    parser.add_argument("--env_name", default=None,
                        help="Override env (defaults to the one in the run)")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--no_render", action="store_true",
                        help="Disable on-screen rendering "
                             "(automatic when --record_video is set)")
    parser.add_argument("--max_episode_steps", type=int, default=0)
    parser.add_argument("--record_video", action="store_true",
                        help="Save mp4 of each episode "
                             "(needs moviepy or imageio[ffmpeg])")
    parser.add_argument("--video_dir", default="",
                        help="Default: <load_dir>/test_videos")
    parser.add_argument("--video_prefix", default="test")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Playback speed multiplier for live render")
    parser.add_argument("--fps_cap", type=float, default=None)
    parser.add_argument("--no_gpu", action="store_true")
    args = parser.parse_args()

    algo = _detect_algo(args.load_dir, override=args.algo)
    if algo not in _LOADERS:
        raise RuntimeError(f"Unknown algo: {algo!r}")

    run_params = _load_run_params(args.load_dir)
    env_name = args.env_name or run_params.get("env_name")
    if env_name is None:
        raise RuntimeError(
            "Cannot determine env_name (not in params.json and not "
            "given via --env_name)."
        )
    run_params["env_name"] = env_name
    run_params["no_gpu"]   = bool(args.no_gpu or run_params.get("no_gpu", False))

    if args.record_video:
        video_dir = args.video_dir or os.path.join(args.load_dir,
                                                   "test_videos")
    else:
        video_dir = None

    env = _make_env(
        env_name,
        render=(not args.no_render) and (not args.record_video),
        record_video=args.record_video,
        video_dir=video_dir,
        video_prefix=args.video_prefix,
        max_episode_steps=args.max_episode_steps,
        prefer_gymnasium=(algo in _PREFER_GYMNASIUM),
    )

    _, act = _LOADERS[algo](env, run_params, args.load_dir)

    step_dt = _get_step_dt(env, env_name)
    if args.record_video or args.no_render or args.speed <= 0:
        target_dt = 0.0
    else:
        target_dt = step_dt / args.speed
        if args.fps_cap is not None and args.fps_cap > 0:
            target_dt = max(target_dt, 1.0 / args.fps_cap)

    print("-" * 50)
    print(f"Algorithm        : {algo.upper()}")
    print(f"Env              : {env_name}")
    print(f"Env step dt      : {step_dt*1000:.2f} ms "
          f"({1/step_dt:.1f} env Hz)")
    if args.max_episode_steps and args.max_episode_steps > 0:
        print(f"Episode limit    : {args.max_episode_steps} steps")
    if args.record_video:
        print(f"Recording videos : ON  -> {video_dir}")
    elif target_dt > 0:
        print(f"Playback target  : {1/target_dt:.1f} Hz "
              f"(speed = {args.speed}x)")
    else:
        print("Playback         : raw simulator speed")
    print("-" * 50)

    total_rewards = []
    for ep in range(args.episodes):
        state = _reset_env(env)
        ep_reward = 0.0
        steps = 0
        done = False
        while not done:
            t0 = time.perf_counter()
            action = act(state)
            state, reward, done, _ = _step_env(env, action)
            if not args.record_video and not args.no_render:
                _try_render(env)
            ep_reward += reward
            steps += 1
            if target_dt > 0:
                sleep_for = target_dt - (time.perf_counter() - t0)
                if sleep_for > 0:
                    time.sleep(sleep_for)
        total_rewards.append(ep_reward)
        print(f"Episode {ep+1}/{args.episodes} | steps: {steps:>4} "
              f"| reward: {ep_reward:>8.2f}")

    print("-" * 50)
    print(f"Average over {args.episodes} episodes: "
          f"{np.mean(total_rewards):.2f} \u00b1 {np.std(total_rewards):.2f}")
    if args.record_video:
        print(f"Videos saved to: {video_dir}")
    env.close()


if __name__ == "__main__":
    play()