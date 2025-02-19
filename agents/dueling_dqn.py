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

from agents.dqn import PrioritizedReplayBuffer, DQNAgent

import copy

class DuelingQNetwork(nn.Module):
    def __init__(
        self, 
        feature_extractor, 
        advantage_network, 
        value_network
        ):
        super(DuelingQNetwork, self).__init__()
        self.feature_extractor = feature_extractor
        self.advantage_network = advantage_network
        self.value_network = value_network
    
    def forward(self, x):
      x = self.feature_extractor(x)
      advantage = self.advantage_network(x)
      value = self.value_network(x)
      q_values = value + (advantage - advantage.mean())
      return q_values

class DuelingDQNAgent(DQNAgent):
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
        cnn_channels=[32, 64, 64],
        cnn_kernel_sizes=[8, 4, 3],
        cnn_strides=[4, 2, 1],
        mlp_feature_extractor_hidden_dims=[128, 128],
    ):
        super().__init__(
            env=env,
            wandb=wandb,
            use_cnn=use_cnn,
            device=device,
            lr=lr,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            tau=tau,
            eps_start=eps_start,
            eps_end=eps_end,
            eps_decay=eps_decay,
            hidden_dims=hidden_dims,
            gradient_clip=gradient_clip,
            double_dqn=double_dqn,
            update_freq=update_freq,
            per_alpha=per_alpha,
            per_beta_start=per_beta_start,
            per_beta_end=per_beta_end,
            per_beta_steps=per_beta_steps,
            cnn_channels=cnn_channels,
            cnn_kernel_sizes=cnn_kernel_sizes,
            cnn_strides=cnn_strides
        )
        self.weight_decay = weight_decay
        self.mlp_feature_extractor_hidden_dims = mlp_feature_extractor_hidden_dims

        # Re-initialize with dueling networks
        self._initialize_model(use_cnn, hidden_dims, cnn_channels, cnn_kernel_sizes, cnn_strides, lr, weight_decay)

    def _initialize_model(self, use_cnn, hidden_dims, cnn_channels, cnn_kernel_sizes, cnn_strides, lr, weight_decay):
        """Override to create a dueling-specific Q network."""
        act_dim = self.env.action_space.n
        if use_cnn:
            self.feature_extractor = CNNBackbone(
                obs_shape=self.env.observation_space.shape,
                output_dim=-1,
                channels=cnn_channels,
                kernel_sizes=cnn_kernel_sizes,
                strides=cnn_strides,
                hidden_dims=[]
            ).to(self.device)
            with torch.no_grad():
                dummy_input = torch.zeros(1, *self.env.observation_space.shape, device=self.device)
                feature_output_dim = self.feature_extractor(dummy_input).view(1, -1).shape[1]
        else:
            self.feature_extractor = MLPBackbone(
                input_dim=self.env.observation_space.shape[0],
                output_dim=self.mlp_feature_extractor_hidden_dims[-1],
                hidden_dims=self.mlp_feature_extractor_hidden_dims[:-1]
            ).to(self.device)
            feature_output_dim = self.mlp_feature_extractor_hidden_dims[-1]

        self.advantage_network = MLPBackbone(
            input_dim=feature_output_dim,
            output_dim=act_dim,
            hidden_dims=hidden_dims
        ).to(self.device)
        self.value_network = MLPBackbone(
            input_dim=feature_output_dim,
            output_dim=1,
            hidden_dims=hidden_dims
        ).to(self.device)
        self.q_network = DuelingQNetwork(
            self.feature_extractor,
            self.advantage_network,
            self.value_network
        ).to(self.device)
        
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=lr, weight_decay=weight_decay)
