import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
from agents.agent import BaseAgent
from models.mlps import MLPBackbone
from models.cnns import CNNBackbone

class PrioritizedReplayBuffer:
    def __init__(self, size, alpha=0.6, beta_start=0.4, beta_end=1.0, beta_steps=100000):
        self.size = size
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.beta_steps = beta_steps
        self.transitions = []
        self.priorities = np.zeros(size, dtype=np.float32)
        self.pos = 0

    def update_beta(self, step):
        """Update beta parameter based on current step"""
        fraction = min(step / self.beta_steps, 1.0)
        self.beta = self.beta_start + fraction * (self.beta_end - self.beta_start)

    def add(self, transition, error=None):
        max_priority = self.priorities[:len(self.transitions)].max() if self.transitions else 1.0
        
        if len(self.transitions) < self.size:
            self.transitions.append(transition)
            self.priorities[len(self.transitions)-1] = max_priority
        else:
            self.transitions[self.pos] = transition
            self.priorities[self.pos] = max_priority
        
        self.pos = (self.pos + 1) % self.size

    def sample(self, batch_size):
        if len(self.transitions) == 0:
            return None, None, None

        N = len(self.transitions)
        if N < batch_size:
            return None, None, None

        # Calculate sampling probabilities
        probs = self.priorities[:N] ** self.alpha
        probs /= probs.sum()

        # Sample indices based on priorities
        indices = np.random.choice(N, batch_size, p=probs)
        
        # Calculate importance sampling weights
        weights = (N * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize weights
        
        transitions = [self.transitions[idx] for idx in indices]
        return transitions, indices, weights

    def update_priorities(self, indices, errors):
        for idx, error in zip(indices, errors.squeeze()):
            self.priorities[idx] = abs(error.item()) + 1e-6
            
    def __len__(self):
        return len(self.transitions)

class DQNAgent(BaseAgent):
    def __init__(
        self, 
        env: gym.Env, 
        wandb=None,
        use_cnn=False,
        device='cpu',
        lr=1e-3,
        weight_decay=1e-4,
        gamma=0.99,
        buffer_size=50000,
        batch_size=128,
        target_update_freq=1,
        tau=0.005,
        eps_start=1.0,
        eps_end=0.02,
        eps_decay=0.995,
        hidden_dims=[128, 128],
        gradient_clip=1.0,
        double_dqn=True,
        update_freq=1,
        per_alpha=0.6,
        per_beta_start=0.4,
        per_beta_end=1.0,
        per_beta_steps=100000,
        # New CNN-specific parameters
        cnn_channels=[32, 64, 64],
        cnn_kernel_sizes=[8, 4, 3],
        cnn_strides=[4, 2, 1]
        ):
        
        super().__init__(env, wandb)
        self.device = device
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.update_freq = update_freq
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.gradient_clip = gradient_clip
        self.double_dqn = double_dqn
        self.total_steps = 0
        self.tau = tau

        # Initialize networks
        self._initialize_model(use_cnn, hidden_dims, cnn_channels, cnn_kernel_sizes, cnn_strides, lr, weight_decay)

        self.replay_buffer = PrioritizedReplayBuffer(
            size=buffer_size,
            alpha=per_alpha,
            beta_start=per_beta_start,
            beta_end=per_beta_end,
            beta_steps=per_beta_steps
        )
      
    def _initialize_model(self, use_cnn, hidden_dims, cnn_channels, cnn_kernel_sizes, cnn_strides, lr, weight_decay):
        """Common model initialization."""
        act_dim = self.env.action_space.n
        if use_cnn:
            self.q_network = CNNBackbone(
                obs_shape=self.env.observation_space.shape,
                output_dim=act_dim,
                hidden_dims=hidden_dims,
                channels=cnn_channels,
                kernel_sizes=cnn_kernel_sizes,
                strides=cnn_strides
            ).to(self.device)
            self.target_network = CNNBackbone(
                obs_shape=self.env.observation_space.shape,
                output_dim=act_dim,
                hidden_dims=hidden_dims,
                channels=cnn_channels,
                kernel_sizes=cnn_kernel_sizes,
                strides=cnn_strides
            ).to(self.device)
        else:
            obs_dim = self.env.observation_space.shape[0]
            self.q_network = MLPBackbone(obs_dim, act_dim, hidden_dims=hidden_dims).to(self.device)
            self.target_network = MLPBackbone(obs_dim, act_dim, hidden_dims=hidden_dims).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=weight_decay)

    def _store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay buffer"""
        transition = (state, action, reward, next_state, done)
        self.replay_buffer.add(transition)

    def _update_policy_step(self):
        """Update policy using prioritized replay"""
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch with priorities
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None

        # Convert batch to tensors
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])
        
        state_batch = torch.FloatTensor(states).to(self.device)
        action_batch = torch.LongTensor(actions).to(self.device)
        reward_batch = torch.FloatTensor(rewards).to(self.device)
        next_state_batch = torch.FloatTensor(next_states).to(self.device)
        done_batch = torch.FloatTensor(dones).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        # Compute current Q values
        q_values = self.q_network(state_batch)
        current_q_values = q_values.gather(1, action_batch.unsqueeze(1))

        # Compute target Q values
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.q_network(next_state_batch).argmax(dim=1, keepdim=True)
                next_q_values = self.target_network(next_state_batch).gather(1, next_actions)
            else:
                next_q_values = self.target_network(next_state_batch).max(1, keepdim=True)[0]

            # Compute target Q values using Bellman equation, (1-done) will be 0 if done=1 True (indicating no future Q values)
            target_q_values = reward_batch.unsqueeze(1) + (1.0 - done_batch.unsqueeze(1)) * self.gamma * next_q_values

        # Compute TD errors for updating priorities
        td_errors = torch.abs(current_q_values - target_q_values).detach().cpu().numpy()
        
        # Update priorities in buffer
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Compute weighted loss
        losses = F.smooth_l1_loss(current_q_values, target_q_values, reduction='none')
        loss = (weights * losses).mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        if self.gradient_clip:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.q_network.parameters(), 
                self.gradient_clip
            )
        
        self.optimizer.step()

        # Update target network if needed
        if self.total_steps % self.target_update_freq == 0:
            # Do a soft update if tau is not 1
            if self.tau == 1:
                self.target_network.load_state_dict(self.q_network.state_dict())
            else:
                for target_param, local_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                    target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

        # Update beta parameter
        self.replay_buffer.update_beta(self.total_steps)

        return {
            "q_loss": loss.item(),
            "mean_q_value": q_values.mean().item(),
            "max_q_value": q_values.max().item(),
            "min_q_value": q_values.min().item(),
            "mean_td_error": td_errors.mean(),
            "max_td_error": td_errors.max(),
            "mean_weight": weights.mean().item()
        }

    def gather_experience(self, env, render=False):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_steps = 0
        metrics_accumulator = {
            "q_loss": [],
            "mean_q_value": [],
            "max_q_value": [],
            "min_q_value": [],
            "mean_td_error": [],
            "max_td_error": [],
            "mean_weight": []
        }

        while not done:
            if render:
                env.render()

            episode_steps += 1
            self.total_steps += 1

            # Select and perform action
            action = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            # Store transition
            self._store_transition(state, action, reward, next_state, done)

            # Update policy if it's time
            if self.total_steps % self.update_freq == 0:
                update_info = self._update_policy_step()
                if update_info:
                    for k, v in update_info.items():
                        metrics_accumulator[k].append(v)

            state = next_state

        # Compute average metrics
        metrics = {
            "total_return": total_reward,
            "steps": episode_steps,
            "total_steps": self.total_steps,
            "eps": self.eps,
            "buffer_size": len(self.replay_buffer),
        }

        # Add averaged update metrics
        for k, v in metrics_accumulator.items():
            if v:  # If any updates occurred
                metrics[k] = np.mean(v)
            else:
                metrics[k] = 0.0

        # Decay epsilon once per episode
        self.eps = max(self.eps_end, self.eps * self.eps_decay)

        return metrics
    
    def select_action(self, state):
        """Select action using epsilon-greedy policy"""
        if random.random() > self.eps:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state)
                return q_values.argmax().item()
        else:
            return random.randrange(self.env.action_space.n)

    def run_episode(self, env, render=False):
        """Run a single episode"""
        return self.gather_experience(env, render)

    def save(self, path):
        """Save the model's state"""
        checkpoint = {
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'eps': self.eps,
            'gamma': self.gamma,
            'gradient_clip': self.gradient_clip,
            'double_dqn': self.double_dqn,
            'device': self.device
        }
        torch.save(checkpoint, path)

    def load(self, path):
        """Load a saved model"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        self.total_steps = checkpoint['total_steps']
        self.eps = checkpoint['eps']
        self.gamma = checkpoint['gamma']
        self.gradient_clip = checkpoint['gradient_clip']
        self.double_dqn = checkpoint['double_dqn']