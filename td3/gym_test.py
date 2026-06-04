import gymnasium as gym

### For testing environments with basic RL
'''
Env List (TD3 needs CONTINUOUS action spaces - marked with *):
Acrobot-v1
CartPole-v1
MountainCar-v0
MountainCarContinuous-v0         *
Pendulum-v0                      *
---Box2D---
CarRacing-v0                     *
LunarLander-v2
LunarLanderContinuous-v2         *
BipedalWalker-v3                 *
---MuJoCo (paper benchmarks)---
HalfCheetah-v2 / v3              *
Hopper-v2 / v3                   *
Walker2d-v2 / v3                 *
Ant-v2 / v3                      *
Reacher-v2                       *
InvertedPendulum-v2              *
InvertedDoublePendulum-v2        *
'''

simulation = 'LunarLanderContinuous-v2'
env = gym.make(simulation)

for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break

env.close()
