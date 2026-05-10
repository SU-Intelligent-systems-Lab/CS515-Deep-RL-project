"""
TD3 agent - Twin Delayed Deep Deterministic Policy Gradient.

Three main contributions vs DDPG:
    1) Clipped Double Q-learning           
    2) Delayed policy updates              
    3) Target policy smoothing             
"""

import copy
import numpy as np
import torch
import torch.nn.functional as F

from td3.model import Actor, Critic
from td3.utils import ReplayBuffer, get_device


class TD3Agent(object):
    def __init__(self, env, agent_params):
        super(TD3Agent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        # dimensions / action scale 
        self.ob_dim     = agent_params['ob_dim']
        self.ac_dim     = agent_params['ac_dim']
        self.max_action = agent_params['max_action']

        # TD3 hyperparameters 
        self.gamma        = agent_params['gamma']        # discount factor
        self.tau          = agent_params['tau']          # target net update rate
        self.policy_noise = agent_params['policy_noise'] # sigma_tilde 
        self.noise_clip   = agent_params['noise_clip']   # c target smoothing cliP
        self.policy_freq  = agent_params['policy_freq']  # delayed updates
        self.batch_size   = agent_params['batch_size']

        self.device = get_device(agent_params.get('no_gpu', False))

        # Actor & Actor target 
        self.actor = Actor(
            self.ob_dim, self.ac_dim, self.max_action,
            n_layers=agent_params['n_layers'],
            size=agent_params['size'],
        ).to(self.device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=agent_params['learning_rate']
        )

        # Twin critic & critic target 
        self.critic = Critic(
            self.ob_dim, self.ac_dim,
            n_layers=agent_params['n_layers'],
            size=agent_params['size'],
        ).to(self.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=agent_params['learning_rate']
        )

        # Replay buffer 
        self.replay_buffer = ReplayBuffer(
            state_dim=self.ob_dim,
            action_dim=self.ac_dim,
            max_size=agent_params.get('buffer_size', int(1e6)),
            device=self.device,
        )
        # number of gradient steps taken 
        self.total_it = 0  

    # Action selection
    def select_action(self, state, noise=0.1):
        """
        Exploration policy: a = clip(pi(s) + N(0, noise*max_action), -max_action, max_action).
        Set noise=0.0 during evaluation (deterministic policy).
        """
        state_t = torch.as_tensor(
            state.reshape(1, -1), dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            action = self.actor(state_t).cpu().numpy().flatten()
        if noise != 0:
            action = action + np.random.normal(
                0.0, self.max_action * noise, size=self.ac_dim
            )
        return action.clip(-self.max_action, self.max_action).astype(np.float32)

    # Training step 
    def train(self):
        """One TD3 update step"""
        self.total_it += 1

        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample mini batch from replay buffer 
        state, action, next_state, reward, not_done = \
            self.replay_buffer.sample(self.batch_size)

        # Compute target y
        with torch.no_grad():
            # (3) Target policy smoothing (eq. 14):
            #     a_next = pi_phi'(s') + clip(N(0, sigma_tilde), -c, c),
            #     then clip to valid action range.
            noise = (
                torch.randn_like(action) * self.policy_noise
            ).clamp(-self.noise_clip, self.noise_clip)

            next_action = (
                self.actor_target(next_state) + noise
            ).clamp(-self.max_action, self.max_action)

            # y = r + gamma * min_i Q_theta_i'(s', a_tilde)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.gamma * target_Q

        # Critic update 
        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + \
                      F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        log = {'critic_loss': critic_loss.item()}

        # Delayed policy + target updates 
        if self.total_it % self.policy_freq == 0:
            # Deterministic policy gradient wrt Q1 only:
            # maximize Q1(s, pi(s))  =>  loss = -mean(Q1)
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Polyak-averaged target network updates 
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                    
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


            log['actor_loss'] = actor_loss.item()

        return log
    
    def add_to_replay_buffer(self, state, action, next_state, reward, done):
        self.replay_buffer.add(state, action, next_state, reward, done)

    def save(self, filename):
        torch.save(self.critic.state_dict(),           filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(),            filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),  filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic",
                                               map_location=self.device))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer", map_location=self.device))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor",
                                              map_location=self.device))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer", map_location=self.device))
        self.actor_target = copy.deepcopy(self.actor)