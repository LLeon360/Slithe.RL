import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from agents.agent import BaseAgent

class ReinforcePolicyGradientsAgent(BaseAgent):
    def __init__(self, env: gym.Env, wandb=None, use_cnn=False, device='cpu', lr=1e-3, gamma=0.99, max_grad_norm=0.5, entropy_coef=1e-4):
        super().__init__(env, wandb)
        self.device = device
        self.gamma = gamma
        self.max_grad_norm = max_grad_norm
        self.entropy_coef = entropy_coef

        if use_cnn:
            in_channels = env.observation_space.shape[0]
            act_dim = env.action_space.n
            self.model = CNNBackbone(obs_shape=env.observation_space.shape, output_dim=act_dim).to(self.device)
        else:
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.n
            self.model = MLPBackbone(obs_dim, act_dim).to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def gather_experience(self, env, render=False):
        obs, _ = env.reset()
        done = False
        rewards, log_probs, entropies = [], [], []

        steps = 0

        while not done:
            if render:
                env.render()
                
            steps += 1

            # Move observation to device and ensure correct shape
            if isinstance(self.model, CNNBackbone):
                # For CNN models 
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            else:
                # For MLP models
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)

            action, log_prob, entropy = self.select_action(obs_tensor)
            obs, reward, terminated, truncated, _ = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            entropies.append(entropy)
            done = terminated or truncated

        return rewards, log_probs, entropies, steps
    
    def update_policy(self, rewards, log_probs, entropies, steps):
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        
        # Move returns to the correct device
        returns = torch.tensor(returns, dtype=torch.float32, device=self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)

        log_probs = torch.stack(log_probs).to(self.device)  # Ensure on the same device
        entropies = torch.stack(entropies).to(self.device)  # Ensure on the same device

        policy_loss = -torch.sum(log_probs * returns)
        entropy_loss = -torch.sum(entropies)
        loss = policy_loss + self.entropy_coef * entropy_loss

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return {
            "total_return": sum(rewards),
            "policy_loss": policy_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": loss.item(),
            "steps": steps
        }

    def select_action(self, obs):
        logits = self.model(obs)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def run_episode(self, env, render=False):
        rewards, log_probs, entropies, steps = self.gather_experience(env, render)
        return self.update_policy(rewards, log_probs, entropies, steps)