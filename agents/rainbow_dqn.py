from agents.dqn import DQNAgent
from agents.dqn import PrioritizedReplayBuffer
from models.cnns import CNNBackbone
from models.mlps import NoisyMLPBackbone
import torch
import torch.nn as nn
import numpy as np
import copy

def compute_conv_output_size(input_size, kernel_size, stride, padding=0):
    """Calculate the output size of a conv layer."""
    return (input_size + 2 * padding - kernel_size) // stride + 1

class C51DuelingQNetwork(nn.Module):
    def __init__(self, obs_shape, output_dim, cnn_channels, cnn_kernel_sizes, cnn_strides, hidden_dims, n_atoms):
        super().__init__()
        self.n_atoms = n_atoms
        self.output_dim = output_dim
        self.v_min = None  # Set later
        self.v_max = None  # Set later
        self.support = None  # Set during forward pass

        # CNN feature extractor
        self.feature_extractor = CNNBackbone(
            obs_shape=obs_shape,  # Pass full shape tuple
            channels=cnn_channels,
            kernel_sizes=cnn_kernel_sizes,
            strides=cnn_strides,
            hidden_dims=[]  # No FC layers here, handled by noisy nets
        )
        
        # Compute feature size analytically 
        input_channels, height, width = obs_shape
        h, w = height, width
        for k, s in zip(cnn_kernel_sizes, cnn_strides):
            h = compute_conv_output_size(h, k, s)
            w = compute_conv_output_size(w, k, s)
        if h <= 0 or w <= 0:
            raise ValueError(f"CNN reduces spatial dimensions to invalid size: {h}x{w}")
        feature_size = cnn_channels[-1] * h * w

        # Noisy networks for value and advantage streams
        self.value_network = NoisyMLPBackbone(
            input_dim=feature_size,
            hidden_dims=hidden_dims,
            output_dim=n_atoms
        )
        self.advantage_network = NoisyMLPBackbone(
            input_dim=feature_size,
            hidden_dims=hidden_dims,
            output_dim=output_dim * n_atoms
        )

    def forward(self, x):
        if self.support is None:
            self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(x.device)
        
        features = self.feature_extractor(x)
        batch_size = features.shape[0]
        
        value_dist = self.value_network(features).view(batch_size, 1, self.n_atoms)
        advantage_dist = self.advantage_network(features).view(batch_size, self.output_dim, self.n_atoms)
        
        q_dist = value_dist + (advantage_dist - advantage_dist.mean(dim=1, keepdim=True))
        q_dist = torch.softmax(q_dist, dim=-1)  # Probabilities over atoms
        
        return q_dist

class RainbowDQNAgent(DQNAgent):
    def __init__(
        self,
        env,
        wandb=None,
        use_cnn=True,
        device='cpu',
        lr=1e-4,
        weight_decay=0,
        gamma=0.99,
        buffer_size=50000,
        batch_size=128,
        target_update_freq=1,
        tau=0.005,
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
        n_steps=3,
        v_min=-10.0,
        v_max=10.0,
        n_atoms=51
    ):
        super().__init__(
            env=env,
            wandb=wandb,
            use_cnn=use_cnn,
            device=device,
            lr=lr,
            weight_decay=weight_decay,
            gamma=gamma,
            buffer_size=buffer_size,
            batch_size=batch_size,
            target_update_freq=target_update_freq,
            tau=tau,
            eps_start=0.0,
            eps_end=0.0,
            eps_decay=0.0,
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
        
        self.n_steps = n_steps
        self.nstep_buffer = []  # store partial transitions # TODO implement n_step
        self.v_min = v_min
        self.v_max = v_max
        self.n_atoms = n_atoms
        self.support = torch.linspace(v_min, v_max, n_atoms).to(self.device)

        if use_cnn:
            obs_shape = env.observation_space.shape  # e.g., (1, 40, 40)
            self.q_network = C51DuelingQNetwork(
                obs_shape=obs_shape,
                output_dim=env.action_space.n,
                cnn_channels=cnn_channels,
                cnn_kernel_sizes=cnn_kernel_sizes,
                cnn_strides=cnn_strides,
                hidden_dims=hidden_dims,
                n_atoms=n_atoms
            ).to(device)
            self.q_network.v_min = v_min
            self.q_network.v_max = v_max

            self.target_network = copy.deepcopy(self.q_network)
            self.target_network.load_state_dict(self.q_network.state_dict())
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise NotImplementedError("RainbowDQN requires CNN backbone")

    def _select_action(self, observation):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
            q_dist = self.q_network(obs_tensor)
            q_values = (q_dist * self.support).sum(dim=-1)
            return q_values.argmax(dim=1).item()

    def _update_policy_step(self):
        # Sample batch with priorities
        batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return None

        # Assuming batch is a list of tuples: (state, action, reward, next_state, done)
        # Convert to NumPy arrays then tensors
        states = np.array([t[0] for t in batch])
        actions = np.array([t[1] for t in batch])
        rewards = np.array([t[2] for t in batch])
        next_states = np.array([t[3] for t in batch])
        dones = np.array([t[4] for t in batch])

        device = self.device
        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).to(device)
        rewards = torch.FloatTensor(rewards).to(device).unsqueeze(1)  # [B,1]
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device).unsqueeze(1)  # [B,1]
        weights = torch.FloatTensor(weights).to(device).unsqueeze(1)  # [B,1]

        delta_z = float(self.v_max - self.v_min) / (self.n_atoms - 1)

        # Compute target distribution and map to fixed support
        with torch.no_grad():
            # Get next-state distributions from target
            next_q_dist = self.target_network(next_states)  # [B, actions, n_atoms]
            if self.double_dqn:
                next_q_dist_online = self.q_network(next_states)  # [B, actions, n_atoms]
                # Sum over support for expected Q values and pick best action per batch element
                next_q_values = (next_q_dist_online * self.support).sum(dim=-1)  # [B, actions]
                next_actions = next_q_values.argmax(dim=-1)  # [B]
                target_dist = next_q_dist[torch.arange(len(next_actions)), next_actions]  # [B, n_atoms]
            else:
                target_dist = next_q_dist.max(dim=1)[0]  # [B, n_atoms]

            # Compute the shifted supports (t_z) for each batch element and each atom:
            # t_z = reward + gamma * support * (1 - done)
            t_z = rewards + (1 - dones) * self.gamma * self.support.unsqueeze(0)  # [B, n_atoms]
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)  # clip to valid range

            # Compute projection of t_z onto the fixed support
            b = (t_z - self.v_min) / delta_z  # [B, n_atoms]
            l = b.floor().long()  # lower indices [B, n_atoms]
            u = b.ceil().long()   # upper indices [B, n_atoms]

            # In case of exact match, floor == ceil; handle by ensuring u = l when b is integer:
            u = torch.where(u < l, l, u)

            # Create an offset to account for batch dimension for vectorized indexing
            batch_size = rewards.shape[0]
            offset = torch.linspace(0, (batch_size - 1) * self.n_atoms, batch_size, device=device).long().unsqueeze(1)  # [B,1]

            # Initialize projected distribution
            proj_dist = torch.zeros((batch_size, self.n_atoms), device=device)

            # Contribution from lower bound
            weight_l = (u.float() - b)  # [B, n_atoms]
            weight_u = (b - l.float())  # [B, n_atoms]

            # Flatten indices for lower and upper contributions
            l_index = (l + offset).view(-1)
            u_index = (u + offset).view(-1)
            # Flatten target distribution (the next state's distribution)
            target_dist_flat = target_dist.view(-1)
            # Accumulate contributions for lower and upper bins
            proj_dist.view(-1).index_add_(0, l_index, (target_dist_flat * weight_l.view(-1)))
            proj_dist.view(-1).index_add_(0, u_index, (target_dist_flat * weight_u.view(-1)))

            # Normalize to get valid probability distribution (adding epsilon to avoid div zero)
            proj_dist = proj_dist / proj_dist.sum(dim=-1, keepdim=True).clamp(min=1e-8)

        # Get predicted distribution for current states (use q_network's forward)
        q_dist = self.q_network(states)  # [B, actions, n_atoms]
        q_dist = q_dist[torch.arange(len(actions)), actions]  # [B, n_atoms]

        # Compute expected Q for current state-action
        predicted_q = (q_dist * self.support).sum(dim=-1)  # [B]

        # Compute cross-entropy loss: using element-wise loss then weighted mean
        log_q = torch.log(q_dist + 1e-8)
        sample_loss = -(proj_dist * log_q).sum(dim=-1)  # [B]
        loss = (sample_loss * weights.squeeze(1)).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.gradient_clip)
        self.optimizer.step()

        # Compute TD errors for priority update
        target_q = (proj_dist * self.support).sum(dim=-1)  # [B]
        td_errors = torch.abs(predicted_q - target_q).detach().cpu().numpy()

        self.replay_buffer.update_priorities(indices, td_errors)

        with torch.no_grad():
            return {
                "q_loss": loss.item(),
                "mean_q_value": predicted_q.mean().item(),
                "max_q_value": predicted_q.max().item(),
                "min_q_value": predicted_q.min().item(),
                "mean_td_error": td_errors.mean(),
                "max_td_error": td_errors.max(),
                "mean_weight": weights.mean().item()
            }
        
    def run_episode(self, env):
        self.q_network.train()
        observation, _ = env.reset()
        done = False
        total_return = 0
        steps = 0
        episode_metrics = {
            "total_return": 0,
            "steps": 0,
            "total_steps": self.total_steps,
            "eps": 0.0,
            "buffer_size": len(self.replay_buffer),
            "q_loss": 0.0,
            "mean_q_value": 0.0,
            "max_q_value": 0.0,
            "min_q_value": 0.0,
            "mean_td_error": 0.0,
            "max_td_error": 0.0,
            "mean_weight": 0.0
        }

        while not done:
            action = self._select_action(observation)
            next_observation, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_return += reward
            steps += 1
            self.total_steps += 1

            # secondary catch for repeated frames, though this should be handled by the env wrapper SkipRedundantFramesWrapper
            if not np.allclose(observation, next_observation) or done:
                self._store_transition(observation, action, reward, next_observation, done)
            else:
                # print(f"DEBUG: No state change at step {steps}, repeated frame")
                pass

            if self.total_steps % self.update_freq == 0 and len(self.replay_buffer) >= self.batch_size:
                metrics = self._update_policy_step()
                episode_metrics.update(metrics)

            if self.total_steps % self.target_update_freq == 0:
                self._soft_update_target_network()

            observation = next_observation

        episode_metrics["total_return"] = total_return
        episode_metrics["steps"] = steps
        episode_metrics["total_steps"] = self.total_steps
        episode_metrics["buffer_size"] = len(self.replay_buffer)
        return episode_metrics

    def _soft_update_target_network(self):
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)