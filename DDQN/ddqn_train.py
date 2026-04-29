import gymnasium as gym
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt

# Import our custom Agent
from agent import DQNAgent

def train_agent(env_name='CartPole-v1', n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, use_ddqn=True):
    """Deep Q-Learning training loop.
    
    Args:
        env_name (str): Name of the Gymnasium environment
        n_episodes (int): Maximum number of training episodes
        max_t (int): Maximum number of timesteps per episode
        eps_start (float): Starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): Minimum value of epsilon
        eps_decay (float): Multiplicative factor (per episode) for decreasing epsilon
        use_ddqn (bool): Toggle between Standard DQN and Double DQN
    """
    print(f"Starting Training on {env_name} | DDQN Enabled: {use_ddqn}")
    
    # Initialize the environment
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # Initialize the agent
    agent = DQNAgent(state_size=state_size, action_size=action_size, seed=0, use_ddqn=use_ddqn)
    
    scores = []                        # List containing scores from each episode
    scores_window = deque(maxlen=100)  # Last 100 scores for checking convergence
    eps = eps_start                    # Initialize epsilon
    
    for i_episode in range(1, n_episodes + 1):
        # Reset environment for the new episode (Gymnasium v26+ API)
        state, info = env.reset()
        score = 0
        
        for t in range(max_t):
            # 1. Agent chooses an action based on current state
            action = agent.act(state, eps)
            
            # 2. Environment responds to the action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 3. Agent learns from this transition
            agent.step(state, action, reward, next_state, done)
            
            # 4. Prepare for the next timestep
            state = next_state
            score += reward
            
            if done:
                break 
                
        # Save most recent score and update epsilon
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps) 
        
        # Print progress to the console
        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}\tEpsilon: {eps:.2f}', end="")
        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_window):.2f}')
            
        # CartPole-v1 is considered "solved" when average score over 100 episodes is >= 475
        if np.mean(scores_window) >= 475.0:
            print(f'\nEnvironment solved in {i_episode - 100:d} episodes!\tAverage Score: {np.mean(scores_window):.2f}')
            # Save the trained model weights
            torch.save(agent.qnetwork_local.state_dict(), f'checkpoint_{"ddqn" if use_ddqn else "dqn"}.pth')
            break
            
    env.close()
    return scores

def plot_scores(scores, title="Training Progress"):
    """Plots the training scores."""
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    # Run a quick dummy test with Double DQN (DDQN)
    # You can change use_ddqn=False to test standard DQN
    scores = train_agent(n_episodes=1000, use_ddqn=True)
    
    # Plot the results
    plot_scores(scores, title="DDQN on CartPole-v1")