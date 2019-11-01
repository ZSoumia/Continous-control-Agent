
# I have chosen to work on DDPG agent

import random
import copy
from collections import namedtuple, deque

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from Actor import Actor
from Critic  import Critic

# Model parameters  most of these values are from Continuous control with deep reinforcement learning paper
# I did some some changes as I was trainning so I changed some values that's why you may find some differences compared with 
# the original paper values 

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4       # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Defining a structure for the reply buffer 

class ReplayBuf:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """ To create a reply buffer 
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.seed = random.seed(seed)
        
        
        self.memory = deque(maxlen=buffer_size) 
        self.action_size = action_size
        self.batch_size = batch_size
        # the following is a structure for the experience 
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        
    def __len__(self):
        """Get the size of the buffer memory"""
        return len(self.memory)  
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Get a random sample to train the network """
        
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)
    
    
    
    
class Ornstein_Uhlenbeck_Noise:
    """Ornstein-Uhlenbeck process.
    Source : https://arxiv.org/pdf/1509.02971.pdf 
    Further details will be found in the report 
    """

    def __init__(self, size, seed, mu=0.0, theta=0.15, sigma=0.2):
        """
        Create a noise object 
        """
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def sample(self):
        
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([np.random.normal(loc=0, scale=1) for _ in range(len(x))])
        self.state = x + dx
        return self.state
    
    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)





class DDPG_Agent():
    
    def __init__(self, state_size, action_size, num_agents):
        """
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of agents in the environment
        """
        random_seed = 1

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.num_agents = num_agents
        
        
        
        # Replay memory
        self.memory = ReplayBuf(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
        
         
        # Noise process
        self.noise = Ornstein_Uhlenbeck_Noise(action_size, random_seed)
        
        # Critic Networks
        self.critic_local = Critic(state_size, action_size, random_seed).to(device)
        self.critic_target = Critic(state_size, action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)


        # Actor Networks
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

       

        

    def step(self, states, actions, rewards, next_states, dones):
        """ add an experience in the reply buffer 
        then sample randomly from that buffer to learn (reason behind the random sampling is to break 
        the correlation between sequential experiences)
        """
        # Save experience 
        for i in range(self.num_agents):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], dones[i])

     
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
        
    def act(self, states, add_noise=True):
        """Returns actions for given state """
        states = torch.from_numpy(states).float().to(device)

        actions = np.zeros((self.num_agents, self.action_size))
        self.actor_local.eval()
        with torch.no_grad():
            for i, state in enumerate(states):
                # Populate list of actions one state at a time
                actions[i, :] = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        
        if add_noise:
            # We add noise for exploration purposes 
            actions += self.noise.sample()
        
        return np.clip(actions, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        
        

        ### Update critic
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Calculate Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)  # adds gradient clipping to stabilize learning
        self.critic_optimizer.step()
        
        ### Update actor 
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        ### Update target networks
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

    def soft_update(self, regular_model, target_model, tau):
        """
            regular_model: it's the most up to date model as it's the one used for trainning 
            target_model:this one is the most stable we copy the weights of the regular model to it 
            tau (float): interpolation parameter 
        """
        for target_param, regular_param in zip(target_model.parameters(),regular_model.parameters()):
            target_param.data.copy_(tau * regular_param.data + (1.0 - tau) * target_param.data)


