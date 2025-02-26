import torch
import torch.nn as nn
from models.noisy_layers import NoisyLinear

class MLPBackbone(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[128, 64]):
        super(MLPBackbone, self).__init__()  # Corrected superclass
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
    
class NoisyMLPBackbone(nn.Module):
    """MLP backbone with NoisyLinear layers for Rainbow DQN."""
    def __init__(self, input_dim, output_dim, hidden_dims=[512]):
        super(NoisyMLPBackbone, self).__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims) - 1):
            layers.append(NoisyLinear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
        layers.append(NoisyLinear(dims[-1], output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

    def reset_noise(self):
        """Reset noise in all NoisyLinear layers."""
        for layer in self.model:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()