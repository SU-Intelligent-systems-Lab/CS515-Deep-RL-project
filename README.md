# CS515 Deep RL Project

Implementations of deep reinforcement learning algorithms: REINFORCE, TD3, DDQN, and PPO.

---

## Performance and Results

### TD3

The TD3 agent was trained to learn locomotion dynamics. Below are the trained behaviors for two distinct continuous control environments.

#### Quadruped Locomotion (Ant)
https://github.com/user-attachments/assets/a72f22af-247c-4c26-9505-5bf836328068

#### Bipedal Locomotion (Walker2d)
https://github.com/user-attachments/assets/f1d9cf69-169f-4caa-b921-5df91f2bf532

---

### PPO — Proximal Policy Optimization

Clean PyTorch implementation of PPO with GAE, trained on MuJoCo continuous-control tasks.

#### Pendulum-v1
Converges to ~-200 (solved threshold) within 1M environment steps.

![PPO Pendulum](ppo/results/ppo_pendulum_training.png)

#### Ant-v4
Reaches ~2200-2500 cumulative reward over 5M steps.

![PPO Ant-v4](ppo/results/ppo_ant_training.png)

#### PPO vs TD3 — Sample Efficiency

![PPO vs TD3](results/ppo_vs_td3_comparison.png)

Both curves show episodic return on a shared environment-steps x-axis.
- **PPO** (blue): running mean of the last 100 episodes, logged every rollout (2048 steps).
- **TD3** (red): per-episode reward smoothed with a trailing rolling mean (window = 40 ep. for Pendulum, 30 for Ant). Shaded band is ±1 std over the same window.

On Pendulum-v1, TD3 converges in ~300K steps vs ~600K for PPO. On Ant-v4, TD3 reaches comparable performance in 1M steps while PPO requires 5M — consistent with the expected off-policy sample-efficiency advantage.

---

## PPO Code Structure

```
ppo/
  config.py          # CLI argument parsing, PPOConfig dataclass
  networks.py        # build_mlp + ActorCritic (discrete & continuous)
  ppo_agent.py       # PPOAgent: act / bootstrap_value / update
  rollout_buffer.py  # fixed-size on-policy buffer, GAE, minibatch iterator
  train.py           # training loop
  logger.py          # TensorBoard wrapper
  ptu.py             # device helpers
  plot_results.py    # regenerate result plots from saved TensorBoard events
  results/           # pre-generated training-curve PNGs
```

## Running PPO

```bash
pip install -r ppo/requirements.txt

# Pendulum-v1
python -m ppo.train --env_name Pendulum-v1 --exp_name ppo_pendulum \
  --total_timesteps 1000000 --n_steps 2048 --entropy_coef 0.0 --target_kl 0.02

# Ant-v4
python -m ppo.train --env_name Ant-v4 --exp_name ppo_ant \
  --total_timesteps 5000000 --n_steps 2048 --size 256 --entropy_coef 0.0 --target_kl 0.02

# Regenerate plots
python ppo/plot_results.py
```

## Key PPO Hyperparameters

| Parameter | Pendulum | Ant-v4 |
|-----------|----------|--------|
| `n_steps` | 2048 | 2048 |
| `update_epochs` | 10 | 10 |
| `minibatch_size` | 64 | 64 |
| `lr` | 3e-4 | 3e-4 |
| `clip_eps` | 0.2 | 0.2 |
| `target_kl` | 0.02 | 0.02 |
| `entropy_coef` | 0.0 | 0.0 |
| `hidden size` | 64 | 256 |
