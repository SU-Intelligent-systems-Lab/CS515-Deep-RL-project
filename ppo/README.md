# PPO — Minimum Viable Implementation

A from-scratch, single-file-per-concept PyTorch implementation of **Proximal Policy
Optimization** (Schulman et al. 2017, [arxiv:1707.06347](https://arxiv.org/abs/1707.06347))
for the CS515 Deep RL term project.

Target: solve `CartPole-v1` end-to-end, then validate on a continuous-control env
(`Pendulum-v1` or `LunarLanderContinuous-v2`).

## Layout

```
ppo/
├── requirements.txt
├── config.py          # argparse CLI
├── ptu.py             # device helpers (ported from ../Deep-RL-intro-main/utils.py)
├── logger.py          # TensorBoard scalar logger (ported, trimmed)
├── networks.py        # build_mlp + ActorCritic (policy + value head)
├── rollout_buffer.py  # on-policy fixed-size buffer
├── ppo_agent.py       # clipped surrogate + value MSE + entropy update
├── train.py           # rollout/update loop (gymnasium 5-tuple API)
└── smoke_test.py      # 1000-step NaN-free sanity check
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r ppo/requirements.txt
```

Python 3.10 or 3.11 recommended (Box2D wheels on Windows).

## Run

Smoke test (verifies the loop runs without NaNs, ~30 s on CPU):
```bash
python -m ppo.smoke_test
```

Train on CartPole (should reach ~500 return within 30–50k steps):
```bash
python -m ppo.train --env_name CartPole-v1 --total_timesteps 50000 --exp_name mvp_cartpole
```

Train on Pendulum (~500k–1M steps, ~1 hr on CPU):
```bash
python -m ppo.train --env_name Pendulum-v1 --total_timesteps 1000000 --entropy_coef 0.0 --exp_name mvp_pendulum
```

TensorBoard:
```bash
tensorboard --logdir ppo/data
```

## What to watch on TensorBoard

- `charts/episodic_return_mean` — should climb monotonically (roughly).
- `ppo/approx_kl` — 0.005–0.02 per update. Spikes > 0.05 mean too many epochs / LR too high.
- `ppo/clip_fraction` — 0.1–0.3. Near 0 = no signal; near 0.5 = ratio exploding.
- `ppo/explained_variance` — climbs toward 0.9+ on CartPole.
- `losses/entropy` — decreases smoothly; must not collapse in the first few k steps.

## Attribution

Infrastructure patterns (`ptu.py` device helpers, `logger.py` TensorBoard logger,
`networks.py` `build_mlp`) are adapted from the `Deep-RL-intro-main` reference
repo that our professor shared (Berkeley DeepRL HW2-style vanilla policy gradient).
The PPO-specific code (rollout buffer with `log_prob_old`/`value`, actor-critic,
clipped surrogate update, training loop) is new to this project.

## Out of scope for MVP (post-MVP extensions)

- GAE(λ) advantage estimation
- Observation normalization (running mean/std)
- Value-function clipping
- Linear LR annealing
- Vectorized envs (`gymnasium.vector.SyncVectorEnv`)
- BipedalWalker / MuJoCo benchmarks
