# CS515 Deep RL Project

Implementations of deep reinforcement learning algorithms: REINFORCE, TD3, DDQN, and PPO.

---

## Performance and Results

The PPO and TD3 agents were trained to learn locomotion dynamics. Below are the trained behaviors for two distinct continuous control environments.

### Quadruped Locomotion (Ant)
#### PPO:
https://github.com/user-attachments/assets/03864d28-b7cd-4300-a662-c4a8a4d959aa
#### TD3:
https://github.com/user-attachments/assets/a72f22af-247c-4c26-9505-5bf836328068

### Bipedal Locomotion (Walker2d)
https://github.com/user-attachments/assets/f1d9cf69-169f-4caa-b921-5df91f2bf532

---
### Pendulum-v1
#### PPO — Proximal Policy Optimization
https://github.com/user-attachments/assets/9da8475b-4fb1-40c9-8cd7-744186ebc4b6

---
### PPO vs TD3 — Sample Efficiency

<img width="2223" height="882" alt="Image" src="https://github.com/user-attachments/assets/2226d8c5-7e1b-4a08-a594-42f27c404f7f" />

Both curves show episodic return on a shared environment-steps x-axis.
- **PPO** (blue): running mean of the last 100 episodes, logged every rollout (2048 steps).
- **TD3** (red): per-episode reward smoothed with a moving mean (window = 40 ep. for Pendulum, 30 for Ant).



## PPO Hyperparameters

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
