# CS515 Deep RL Project
Implementations of deep reinforcement learning algorithms: **REINFORCE**, ***TRPO**, **PPO**, **TD3**, and **DDQN**.

| Algorithm                                   | Action space | Folder       |
|---------------------------------------------|--------------|--------------|
| **TD3** — Twin Delayed DDPG                 | continuous   | `td3/`       |
| **PPO** — Proximal Policy Optimisation      | both         | `ppo/`       |
| **TRPO** — Trust Region Policy Optimisation | both         | `trpo/`      |
| **REINFORCE** with value baseline           | both         | `reinforce/` |
| **DDQN** — Double Deep Q-Network            | discrete     | `DDQN/`      |

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

---

## Multi-Seed Benchmark

To make the comparison more reliable, we re-ran every (algorithm,
environment) combination with 5 random seeds. Every 5 to 10k training
steps we pause training, freeze the policy, and run 10 evaluation
episodes with deterministic actions:

- mean of the Gaussian for PPO, TRPO and REINFORCE
- actor output with zero noise for TD3
- argmax over Q values for DDQN

The plots below show the mean across the 5 seeds with shaded bands at
one standard deviation.

### Learning curves on continuous control

<p align="center">
  <img src="https://github.com/user-attachments/assets/7822fea5-7dd7-48ba-9469-3d0dea1d9fd0" width="49%" />
  <img src="https://github.com/user-attachments/assets/fffc2a9a-2340-469e-8187-d91496258532" width="49%" />
  <img src="https://github.com/user-attachments/assets/fdaa23ce-68a1-4634-8892-175c72b3d600" width="49%" />
  <img src="https://github.com/user-attachments/assets/3bda2a5a-c230-4fa4-b0c9-a11b05dc96c8" width="49%" />
</p>




### Final returns (continuous control)

Mean ± std over 5 seeds, computed from the last 10% of evaluation
points. Bold is the best per row.

| Environment      | REINFORCE        | TRPO             | PPO                  | TD3                  |
|------------------|------------------|------------------|----------------------|----------------------|
| Pendulum-v1      | -1071 ± 11       | **-124 ± 8**     | -130 ± 11            | -148 ± 5             |
| Hopper-v4        | 339 ± 53         | 2666 ± 793       | **3100 ± 356**       | 2808 ± 874           |
| HalfCheetah-v4   | 397 ± 324        | 1138 ± 73        | 2912 ± 1192          | **9794 ± 913**       |
| Walker2d-v4      | 261 ± 27         | 1343 ± 694       | 2827 ± 438           | **3444 ± 1267**      |

### Learning curves on discrete control (LunarLander-v2)

<img width="1456" height="532" alt="learning_curves_discrete" src="https://github.com/user-attachments/assets/c2b6c8d0-e927-495b-b4eb-f151aee3d0f3" />


### Final returns (discrete control)

Mean ± std over 5 seeds, computed from the last 10% of evaluation
points. Bold is the best per row.

| Environment      | REINFORCE        | TRPO             | PPO              | DDQN             |
|------------------|------------------|------------------|------------------|------------------|
| LunarLander-v2   | -394 ± 145       | 236 ± 29         | 206 ± 65         | **236 ± 27**     |

### Wall-clock cost

<img width="728" height="532" alt="wallclock_hopper-v4_log" src="https://github.com/user-attachments/assets/271f3e43-7cc7-4bd0-818b-46682459f4da" />


On Hopper-v4 the same 1M environment-step budget takes around 1.5k
wall-clock seconds for PPO and around 12k seconds for TD3.

### Aggregate normalized score

<img width="980" height="560" alt="aggregate_normalised_continuous" src="https://github.com/user-attachments/assets/32108aeb-3cb2-4188-9490-e4f9a83fbe1a" />
<img width="980" height="560" alt="aggregate_normalised_discrete" src="https://github.com/user-attachments/assets/fd4bf813-826b-4858-bc9b-862fd075eb95" />

Each algorithm's final return per environment is normalized so the
best algorithm scores 1.0 and the worst 0.0. The bars are the average
across the continuous environments.

The .npy arrays underlying every plot above are saved under `results/`.

## Hyperparameters
*The hyperparameters for each algorithm are in `<algo>/config.json` and follow the original paper defaults and the Stable-Baselines3 RL Zoo conventions. Use `--set key=value` to override any field on the command line (see §3.3).*


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
``` text
.
├── README.md
├── DDQN
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
```  

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

`<ALGO>` could be one of the following `{td3, ppo, reinforce, ddqn, trpo}`. Hyperparameters come from
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

# TRPO — both action spaces
python main.py --algo trpo --env_name CartPole-v1
python main.py --algo trpo --env_name Pendulum-v1 --seed 1

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
| `--mode`          | select the mode {train, test} (default, train)               | 


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
| `model_final.pt`                                      | PPO / REINFORCE / TRPO | torch state dict               |
| `model.pth`                                           | DDQN                | torch state dict                  |

The model is **always saved** at the end of training, regardless of
whether an environment-specific solve threshold was reached.

---
