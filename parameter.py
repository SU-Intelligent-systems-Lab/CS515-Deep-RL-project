import argparse
import json
import os
import time
from typing import Any, Dict
 
 
VALID_ALGOS = ("td3", "ppo", "reinforce", "ddqn", "trpo")
 
 
def _coerce(s: str) -> Any:
    sl = s.lower()
    if sl in ("true", "false"):
        return sl == "true"
    if sl in ("null", "none"):
        return None
    try:
        if "." in s or "e" in sl:
            return float(s)
        return int(s)
    except ValueError:
        return s
 
 
def _apply_env_overrides(cfg: Dict[str, Any], env_name: str) -> Dict[str, Any]:
    if "defaults" in cfg:
        base = dict(cfg["defaults"])
    else:
        base = {k: v for k, v in cfg.items() if k != "env_overrides"}
    env_overrides = cfg.get("env_overrides", {}) or {}
    if env_name in env_overrides:
        base.update(env_overrides[env_name])
    return base
 
 
def _build_logdir(logroot: str, algo: str, env_name: str,
                  exp_name: str) -> str:
    if not os.path.isabs(logroot):
        logroot = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               logroot)
    ts = time.strftime("%d-%m-%Y_%H-%M-%S")
    name = f"{exp_name or algo}_{env_name}_{ts}"
    logdir = os.path.join(logroot, algo, name)
    os.makedirs(logdir, exist_ok=True)
    return os.path.abspath(logdir)
 
 
def parse_args() -> Dict[str, Any]:
    parser = argparse.ArgumentParser(
        description="Unified DRL trainer entry point.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
 
    parser.add_argument("--algo", type=str, required=True, choices=VALID_ALGOS)
    parser.add_argument("--env_name", type=str, required=True)
 
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--seed",     type=int, default=None)
    parser.add_argument("--config",   type=str, default=None,
                        help="Path to a custom JSON config "
                             "(default: <algo>/config.json)")
    parser.add_argument("--logroot",  type=str, default="runs")
    parser.add_argument("--no_gpu",   action="store_true")
 
    parser.add_argument("--load_model", type=str, default="")
 
    parser.add_argument("--render",      action="store_true")
    parser.add_argument("--render_freq", type=int, default=None)
    parser.add_argument("--plot",      dest="plot",
                        action="store_true",  default=None)
    parser.add_argument("--no_plot",   dest="plot",      action="store_false")
    parser.add_argument("--show_plot", dest="show_plot",
                        action="store_true",  default=None)
    parser.add_argument("--no_show_plot", dest="show_plot",
                        action="store_false")
 
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VAL",
                        help="Override any config field, e.g. "
                             "--set lr=1e-4 batch_size=128 target_kl=null")
 
    args = parser.parse_args()
 
    code_dir = os.path.dirname(os.path.realpath(__file__))
    config_path = args.config or os.path.join(code_dir, args.algo,
                                              "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(
            f"\n[parameter] Could not find config for algo "
            f"'{args.algo}': {config_path}\n"
            f"             Create it or pass --config <path>."
        )
    with open(config_path, "r") as f:
        raw_cfg = json.load(f)
    params = _apply_env_overrides(raw_cfg, args.env_name)
 
    params["algo"]     = args.algo
    params["env_name"] = args.env_name
    params["no_gpu"]   = bool(args.no_gpu or params.get("no_gpu", False))
 
    if args.exp_name is not None:
        params["exp_name"] = args.exp_name
    params.setdefault("exp_name", args.algo)
 
    if args.seed is not None:
        params["seed"] = args.seed
    params.setdefault("seed", 0)
 
    if args.load_model:
        params["load_model"] = args.load_model
    params.setdefault("load_model", "")
 
    if args.render:
        params["render"] = True
    params.setdefault("render", False)
 
    if args.render_freq is not None:
        params["render_freq"] = args.render_freq
    params.setdefault("render_freq", 1)
 
    if args.plot is not None:
        params["plot"] = args.plot
    params.setdefault("plot", True)
 
    if args.show_plot is not None:
        params["show_plot"] = args.show_plot
    params.setdefault("show_plot", True)
 
    for kv in args.set:
        if "=" not in kv:
            raise ValueError(f"--set expects KEY=VAL, got: {kv!r}")
        k, v = kv.split("=", 1)
        params[k.strip()] = _coerce(v.strip())
 
    params["logdir"] = _build_logdir(args.logroot, args.algo,
                                     args.env_name, params.get("exp_name"))
 
    with open(os.path.join(params["logdir"], "algo.txt"), "w") as f:
        f.write(args.algo + "\n")
 
    serialisable = {k: v for k, v in params.items()
                    if isinstance(v, (int, float, str, bool, list,
                                      type(None)))}
    with open(os.path.join(params["logdir"], "params.json"), "w") as f:
        json.dump(serialisable, f, indent=2)
 
    return params
 

if __name__ == "__main__":
    cfg = parse_args()
    print(json.dumps({k: v for k, v in cfg.items()
                      if isinstance(v, (int, float, str, bool, list,
                                        type(None)))},
                     indent=2))