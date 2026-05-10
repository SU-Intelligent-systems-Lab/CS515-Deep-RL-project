<img width="1145" height="361" alt="image" src="https://github.com/user-attachments/assets/36d13095-6417-4c55-b00b-efef9b86cf07" /># CS515 Deep RL Project

Implementations of deep reinforcement learning algorithms: **REINFORCE**, **TD3**,
**DDQN**, and **PPO**.

| Algorithm                                  | Action space | Folder       |
|--------------------------------------------|--------------|--------------|
| **TD3** — Twin Delayed DDPG                | continuous   | `td3/`       |
| **PPO** — Proximal Policy Optimisation     | both         | `ppo/`       |
| **REINFORCE** with value baseline          | both         | `reinforce/` |
| **DDQN** — Double Deep Q-Network           | discrete     | `ddqn/`      |

---

## Performance and Results

The PPO and TD3 agents were trained to learn locomotion dynamics. Below are
the trained behaviors for two distinct continuous control environments.


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


## 1. Setup

### 1.1 Create an environment

Conda (recommended):
```bash
conda create -n drl python=3.10 -y
conda activate drl
```

### 1.2 Install Python dependencies

### 1.3 Box2D environments

`LunarLander`, `LunarLanderContinuous`, `BipedalWalker`, and `CarRacing`
require `box2d-py`, which will be installed using conda:

```bash
conda install -c conda-forge box2d-py swig
```

```bash
pip install -r requirements.txt
```

---

## 2. Repository structure
''' text
.
├── README.md
├── ddqn
│   ├── config.json
│   ├── config.py
│   ├── ddqn_agent.py
│   ├── ddqn_train.py
│   ├── dqn_vs_ddqn_result.png
│   ├── networks.py
│   └── replay_buffer.py
├── main.py
├── parameter.py
├── ppo
│   ├── README.md
│   ├── __init__.py
│   ├── config.json
│   ├── config.py
│   ├── logger.py
│   ├── networks.py
│   ├── plot_results.py
│   ├── ppo_agent.py
│   ├── ptu.py
│   ├── results
│   │   ├── ppo_ant_training.png
│   │   └── ppo_pendulum_training.png
│   ├── rollout_buffer.py
│   ├── smoke_test.py
│   └── train.py
├── reinforce
│   ├── __init__.py
│   ├── config.json
│   ├── config.py
│   ├── logger.py
│   ├── networks.py
│   ├── ptu.py
│   ├── reinforce_agent.py
│   ├── render.py
│   └── train.py
├── requirements.txt
├── td3
│   ├── agent.py
│   ├── config.json
│   ├── config.py
│   ├── gym_test.py
│   ├── model.py
│   ├── results
│   │   ├── ant_training_curves.png
│   │   ├── pendulum_training_curveps.png
│   │   └── walker_runner_training_curves.png
│   ├── train.py
│   └── utils.py
├── test.py
└── trpo
    ├── __init__.py
    ├── config.json
    ├── config.py
    ├── conjugate_gradient.py
    ├── logger.py
    ├── networks.py
    ├── ptu.py
    ├── rollout_buffer.py
    ├── train.py
    └── trpo_agent.py
'''    

Every run folder under `runs/<algo>/` contains:
- `algo.txt`, `params.json` — written by `parameter.py` in order for `test.py` to
  auto detect the algorithm and rebuild the agent;
- training history (`episode_rewards.csv`, `eval_rewards.csv`,
  `episode_scores.csv`, ...);
- `summary.json` and `training_curves.png`.

---

## 3. Training

### 3.1 The command

```bash
python main.py --algo <ALGO> --env_name <ENV> [overrides]
```

`<ALGO>` could be of of the following `{td3, ppo, reinforce, ddqn, trpo}`. Hyperparameters come from
`<algo>/config.json`. The JSON supports an `env_overrides` block that
adjusts hyperparameters per environment. Anything in JSON can
be overridden on the command line via `--set key=value`.

### 3.2 Examples per algorithm

```bash
# TD3 — continuous control
python main.py --algo td3 --mode train --env_name Pendulum-v1
python main.py --algo td3 --env_name LunarLanderContinuous-v2 --render
python main.py --algo td3 --env_name HalfCheetah-v4 --seed 7

# PPO — both action spaces
python main.py --algo ppo --env_name CartPole-v1
python main.py --algo ppo --env_name Pendulum-v1
python main.py --algo ppo --env_name Ant-v4 \
                          --set entropy_coef=0.0 lr=1e-4

# REINFORCE — both action spaces
python main.py --algo reinforce --env_name CartPole-v1
python main.py --algo reinforce --env_name Pendulum-v1 --seed 1

# Double DQN — discrete control
python main.py --algo ddqn --env_name CartPole-v1
python main.py --algo ddqn --env_name LunarLander-v2 --seed 0
```

### 3.3 Common command-line flags

| Flag             | Meaning                                                       |
|------------------|----------------------------------------------------------------|
| `--seed N`       | RNG seed (default from JSON; usually `0` or `1`)              |
| `--exp_name S`   | Custom run name prefix (default: the algo name)               |
| `--no_gpu`       | Force CPU                         |
| `--render`       | TD3 only: render the env during training                      |
| `--load_model P` | TD3 only: resume from a checkpoint prefix                     |
| `--no_plot`      | Skip the final `training_curves.png`                          |
| `--no_show_plot` | Save the plot but don't pop up a window                       |
| `--set k=v ...`  | Override any field in `<algo>/config.json`                    |
| `--config PATH`  | Use a custom config JSON instead of `<algo>/config.json`      |
| `--logroot DIR`  | Root for all run folders (default `runs/`)                    |
| `--mode`          | select the mode {train, test} (defualt, train)               | 


---

## 4. Testing a trained agent

To see the trained agent in action you could run it using test.py file

```bash
# Live rendering, 5 episodes (default)
python test.py --load_dir runs/td3/td3_Pendulum-v1_28-04-2026_04-41-52      or
python main.py --mode test --load_dir runs/td3/td3_Pendulum-v1_28-04-2026_04-41-52 

# Record mp4s to <load_dir>/test_videos/
python test.py --load_dir runs/td3/td3_Ant-v4_... \
               --record_video --episodes 3 --max_episode_steps 300

# Headless evaluation
python test.py --load_dir runs/ppo/ppo_CartPole-v1_... --no_render
```

The script auto detects the algorithm from `<load_dir>/algo.txt` and
rebuilds the matching agent from the run's `params.json`.

---

## 5. Outputs

Every run folder contains:

| File                                                  | Created by          | Contents                          |
|-------------------------------------------------------|---------------------|-----------------------------------|
| `algo.txt`                                            | `parameter.py`      | algorithm name, for `test.py`     |
| `params.json`                                         | `parameter.py`      | resolved hyperparameters          |
| `summary.json`                                        | per-algo `train.py` | run summary stats                 |
| `training_curves.png`                                 | per-algo `train.py` | reward curve(s)                   |
| `episode_rewards.csv`                                 | TD3                 | per-episode rewards + step counts |
| `eval_rewards.csv`                                    | TD3                 | periodic eval returns             |
| `episode_scores.csv`                                  | DDQN                | per-episode scores                |
| `model_actor`, `model_critic`, `model_*_optimizer`    | TD3                 | torch state dicts                 |
| `model_final.pt`                                      | PPO / REINFORCE     | torch state dict                  |
| `model.pth`                                           | DDQN                | torch state dict                  |

The model is **always saved** at the end of training, regardless of
whether an environment-specific solve threshold was reached.

---
