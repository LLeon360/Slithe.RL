import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
import gymnasium as gym
import abc

from models.mlps import MLPBackbone
from models.cnns import CNNBackbone

class BaseAgent(metaclass=abc.ABCMeta):
    def __init__(self, env: gym.Env, wandb=None):
        self.env = env
        self.wandb = wandb
        pass

    @abc.abstractmethod
    def gather_experience(self, env, render=False):
        pass

    @abc.abstractmethod
    def update_policy(self, rewards, log_probs, entropies):
        pass

    @abc.abstractmethod
    def select_action(self, obs: np.ndarray):
        pass

    @abc.abstractmethod
    def run_episode(self, env, render=False):
        pass
