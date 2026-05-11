import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

# Import local modules
from DDQN.networks import QNetwork
from DDQN.replay_buffer import ReplayBuffer

# Hyperparameters
BUFFER_SIZE = int(1e5)  # Replay buffer size
BATCH_SIZE = 64         # Minibatch size
GAMMA = 0.99            # Discount factor
TAU = 5e-3              # For soft update of target parameters
LR = 1e-3               # Learning rate 
UPDATE_EVERY = 1        # How often to update the network

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DQNAgent():
    """Interacts with and learns from the environment using DQN or DDQN."""

    def __init__(self, state_size, action_size, seed, use_ddqn=False):
        """Initialize an Agent object.
        
        Args:
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            use_ddqn (bool): Whether to use Double DQN instead of standard DQN
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.use_ddqn = use_ddqn

        # Initialize Q-Networks (Local and Target)
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Args:
            state (array_like): Current state
            eps (float): Epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        
        # Set local network to evaluation mode
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        # Revert to training mode
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy()) # Exploitation
        else:
            return random.choice(np.arange(self.action_size))  # Exploration

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.
        
        Args:
            experiences (Tuple[torch.Tensor]): Tuple of (s, a, r, s', done) tuples 
            gamma (float): Discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if self.use_ddqn:
            # --- DOUBLE DQN (DDQN) LOGIC ---
            # 1. Get best actions for the next states from the local network
            best_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)
            # 2. Evaluate these best actions using the target network
            Q_targets_next = self.qnetwork_target(next_states).gather(1, best_actions).detach()
        else:
            # --- STANDARD DQN LOGIC ---
            # Get max predicted Q values for the next states from the target model
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss (Mean Squared Error)
        loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network parameters
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Args:
            local_model (PyTorch model): Weights will be copied from
            target_model (PyTorch model): Weights will be copied to
            tau (float): Interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)