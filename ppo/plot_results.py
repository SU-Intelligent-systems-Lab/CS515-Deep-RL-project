"""
plot_results.py  --  PPO training curves
=========================================
Generates result plots for the PPO runs:
  - Pendulum-v1: episode return vs environment steps
  - Ant-v4:      episode return + KL / clip-fraction diagnostics

Usage (from CS515-Deep-RL/ root):
    python ppo/plot_results.py

Output (written to ppo/results/):
    ppo_pendulum_training.png
    ppo_ant_training.png
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "ppo" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pick the latest matching run directory
PENDULUM_GLOB = "ppo/data/ppo_pendulum_rerun_Pendulum*"
ANT_GLOB      = "ppo/data/ppo_ant_v2_Ant-v4_2026-04-29*"

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────
PPO_COLOR = "#1A56DB"

plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     11,
    "axes.linewidth":     0.9,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#cccccc",
    "grid.color":         "#e4e4e4",
    "grid.linewidth":     0.6,
    "figure.facecolor":   "white",
    "axes.facecolor":     "#fafafa",
})

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def resolve_newest(pattern: str) -> Path | None:
    matches = sorted(glob.glob(str(REPO_ROOT / pattern)))
    if not matches:
        print(f"  [warn] no match for {pattern}")
        return None
    return Path(matches[-1])


def load_scalar(run_dir: Path, tag: str):
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    evs = ea.Scalars(tag)
    return np.array([e.step for e in evs]), np.array([e.value for e in evs])


def fmt_steps(ax) -> None:
    def _f(x, _):
        if x == 0:   return "0"
        if x >= 1e6: return f"{x/1e6:.0f}M" if (x/1e6) == int(x/1e6) else f"{x/1e6:.1f}M"
        return f"{int(x/1e3)}K"
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_f))


def ema(vals: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    out = np.empty_like(vals, dtype=float)
    out[0] = vals[0]
    for i in range(1, len(vals)):
        out[i] = alpha * vals[i] + (1.0 - alpha) * out[i - 1]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Plot 1: Pendulum
# ─────────────────────────────────────────────────────────────────────────────

def plot_pendulum() -> None:
    rd = resolve_newest(PENDULUM_GLOB)
    if rd is None:
        return

    steps, vals = load_scalar(rd, "charts/episodic_return_mean")
    if len(steps) == 0:
        print("  [warn] no return data for Pendulum run")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(steps, vals, color=PPO_COLOR, linewidth=2.0, label="PPO (mean of last 100 ep.)")
    ax.axhline(-200, color="#888888", linewidth=1.0, linestyle="--", label="Solved threshold (−200)")

    ax.set_title("PPO on Pendulum-v1", fontweight="bold", pad=9)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.set_ylim(-1650, 80)
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", zorder=0)
    fmt_steps(ax)

    plt.tight_layout()
    out = RESULTS_DIR / "ppo_pendulum_training.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Plot 2: Ant-v4  (3-panel: return / KL / clip fraction)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ant() -> None:
    rd = resolve_newest(ANT_GLOB)
    if rd is None:
        return

    steps_ret, vals_ret = load_scalar(rd, "charts/episodic_return_mean")
    if len(steps_ret) == 0:
        print("  [warn] no return data for Ant run")
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(steps_ret, vals_ret, color=PPO_COLOR, linewidth=1.5, alpha=0.35)
    ax.plot(steps_ret, ema(vals_ret, 0.08), color=PPO_COLOR, linewidth=2.0,
            label="PPO (mean of last 100 ep.)")

    ax.set_title("PPO on Ant-v4", fontweight="bold", pad=9)
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel("Episode Return")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", zorder=0)
    fmt_steps(ax)

    plt.tight_layout()
    out = RESULTS_DIR / "ppo_ant_training.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"  Saved -> {out}")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Generating PPO result plots...")
    plot_pendulum()
    plot_ant()
    print("Done.")


if __name__ == "__main__":
    main()
