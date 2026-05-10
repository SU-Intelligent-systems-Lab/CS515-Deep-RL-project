"""
Unified entry point for all DRL algorithms - training AND testing.
 
Modes
-----
    --mode train (default)  Train a fresh model
    --mode test             Load an existing checkpoint and play / record it
 
Examples
--------
    # Training
    python main.py --algo td3 --env_name Pendulum-v1
    python main.py --mode train --algo ppo --env_name CartPole-v1 --seed 7
    python main.py --algo ppo --env_name Pendulum-v1 --set entropy_coef=0.0
 
    # Testing (auto-detects algo from <load_dir>/algo.txt)
    python main.py --mode test --load_dir runs/td3/td3_Pendulum-v1_28-04-2026_04-41-52
    python main.py --mode test --load_dir <run> --record_video --episodes 3
 
Help:
    python main.py --help              # training help (default mode)
    python main.py --mode test --help  # testing help
"""

import sys
import traceback
import argparse

from parameter import parse_args


def _split_mode():
    """First-pass parser: peel off --mode without consuming other flags.
 
    Uses parse_known_args + add_help=False so that --help is forwarded to the
    appropriate downstream parser (training or testing) instead of being
    intercepted here.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument(
        "--mode",
        choices=("train", "test"),
        default="train",
        help="train a new model (default) or test an existing one",
    )
    args, remaining = pre.parse_known_args()
    return args.mode, remaining


# Per-algo dispatchers                                                        
def _dispatch_td3(params):
    from td3.config import TD3Config
    from td3.train  import run
 
    cfg = TD3Config(
        env_name        = params["env_name"],
        exp_name        = params["exp_name"],
        seed            = int(params["seed"]),
        max_timesteps   = int(params["max_timesteps"]),
        start_timesteps = int(params["start_timesteps"]),
        eval_freq       = int(params["eval_freq"]),
        eval_episodes   = int(params["eval_episodes"]),
        batch_size      = int(params["batch_size"]),
        discount        = float(params["discount"]),
        tau             = float(params["tau"]),
        policy_noise    = float(params["policy_noise"]),
        noise_clip      = float(params["noise_clip"]),
        policy_freq     = int(params["policy_freq"]),
        expl_noise      = float(params["expl_noise"]),
        n_layers        = int(params["n_layers"]),
        size            = int(params["size"]),
        learning_rate   = float(params["learning_rate"]),
        buffer_size     = int(params["buffer_size"]),
        save_model      = bool(params.get("save_model", True)),
        load_model      = str(params.get("load_model", "")),
        render          = bool(params.get("render", False)),
        render_freq     = int(params.get("render_freq", 1)),
        plot            = bool(params.get("plot", True)),
        show_plot       = bool(params.get("show_plot", True)),
        log_dir         = params["logdir"],
        no_gpu          = bool(params["no_gpu"]),
    )
    run(cfg)


def _dispatch_ppo(params):
    """Build a PPOConfig from the merged JSON+CLI params, then run()."""
    from ppo.config import PPOConfig
    from ppo.train import run

    cfg = PPOConfig(
        env_name        = params["env_name"],
        exp_name        = params["exp_name"],
        seed            = int(params["seed"]),
        total_timesteps = int(params["total_timesteps"]),
        n_steps         = int(params["n_steps"]),
        minibatch_size  = int(params["minibatch_size"]),
        update_epochs   = int(params["update_epochs"]),
        lr              = float(params["lr"]),
        gamma           = float(params["gamma"]),
        gae_lambda      = float(params["gae_lambda"]),
        clip_eps        = float(params["clip_eps"]),
        target_kl       = (None if params.get("target_kl") is None
                           else float(params["target_kl"])),
        value_coef      = float(params["value_coef"]),
        entropy_coef    = float(params["entropy_coef"]),
        max_grad_norm   = float(params["max_grad_norm"]),
        n_layers        = int(params["n_layers"]),
        size            = int(params["size"]),
        log_dir         = params["logdir"],     # use OUR unified logdir
        no_gpu          = bool(params["no_gpu"]),
    )
    run(cfg)


def _dispatch_reinforce(params):
    from reinforce.config import REINFORCEConfig
    from reinforce.train import run

    cfg = REINFORCEConfig(
        env_name                = params["env_name"],
        exp_name                = params["exp_name"],
        seed                    = int(params["seed"]),
        total_episodes          = int(params["total_episodes"]),
        min_episodes_per_update = int(params["min_episodes_per_update"]),
        lr                      = float(params["lr"]),
        gamma                   = float(params["gamma"]),
        entropy_coef            = float(params["entropy_coef"]),
        n_layers                = int(params["n_layers"]),
        size                    = int(params["size"]),
        log_dir                 = params["logdir"],
        no_gpu                  = bool(params["no_gpu"]),
    )
    run(cfg)


def _dispatch_ddqn(params):
    from ddqn.config     import DDQNConfig
    from ddqn.ddqn_train import run
 
    cfg = DDQNConfig(
        env_name        = params["env_name"],
        exp_name        = params["exp_name"],
        seed            = int(params["seed"]),
        n_episodes      = int(params["n_episodes"]),
        max_t           = int(params["max_t"]),
        eps_start       = float(params["eps_start"]),
        eps_end         = float(params["eps_end"]),
        eps_decay       = float(params["eps_decay"]),
        use_ddqn        = bool(params["use_ddqn"]),
        score_threshold = (None if params.get("score_threshold") is None
                           else float(params["score_threshold"])),
        plot            = bool(params.get("plot", True)),
        show_plot       = bool(params.get("show_plot", True)),
        log_dir         = params["logdir"],
        no_gpu          = bool(params["no_gpu"]),
    )
    run(cfg)


_DISPATCH = {
    "td3":       _dispatch_td3,
    "ppo":       _dispatch_ppo,
    "reinforce": _dispatch_reinforce,
    "ddqn":       _dispatch_ddqn,
}

# Main                                                                        
def _print_header(params):
    bar = "=" * 60
    print(bar)
    print(f"  Algorithm   : {params['algo'].upper()}")
    print(f"  Environment : {params['env_name']}")
    print(f"  Seed        : {params['seed']}")
    print(f"  Exp name    : {params['exp_name']}")
    print(f"  Logdir      : {params['logdir']}")
    if params.get("load_model"):
        print(f"  Resume from : {params['load_model']}")
    print(bar)

def _run_training():
    params = parse_args()
    _print_header(params)
 
    algo = params["algo"]
    if algo not in _DISPATCH:
        print(f"[main] ERROR: no dispatcher registered for algo '{algo}'")
        sys.exit(1)
 
    try:
        _DISPATCH[algo](params)
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user (Ctrl+C).")
    except Exception:
        print("\n[main] Training crashed:")
        traceback.print_exc()
        sys.exit(1)
 
 
def _run_testing():
    # Local import: only paid when actually testing, and avoids any chance of
    # circular import at module load time.
    from test import play
    try:
        play()
    except KeyboardInterrupt:
        print("\n[main] Interrupted by user (Ctrl+C).")
    except SystemExit:
        # argparse inside play() may sys.exit on --help; let it.
        raise
    except Exception:
        print("\n[main] Test run crashed:")
        traceback.print_exc()
        sys.exit(1)
 
 
# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    mode, remaining = _split_mode()
    # Hand the rest of the CLI to whichever parser actually handles it.
    sys.argv = [sys.argv[0]] + remaining
 
    if mode == "test":
        _run_testing()
    else:
        _run_training()
 
 
if __name__ == "__main__":
    main()