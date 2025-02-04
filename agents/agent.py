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

    @abc.abstractmethod
    def run_episode(self, env, render=False):
        """Run a single episode and return metrics"""
        pass

    @abc.abstractmethod
    def select_action(self, obs: np.ndarray):
        """Select an action given an observation"""
        pass

    # Optionally keep these for saving/loading
    @abc.abstractmethod
    def save(self, path):
        pass

    @abc.abstractmethod
    def load(self, path):
        pass