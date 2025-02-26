#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Configuration Section
import os
import datetime
import json
from pathlib import Path
import gymnasium as gym
import ale_py
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation, RecordVideo, TransformReward
from agents.reinforce import ReinforcePolicyGradientsAgent
from agents.dqn import DQNAgent
from agents.dueling_dqn import DuelingDQNAgent
from agents.rainbow_dqn import RainbowDQNAgent
from models.cnns import CNNBackbone
from wrappers.gym_wrappers import ProgressiveRewardWrapper, ChannelFirstWrapper, NormalizeWrapper, ChannelWiseFrameStack, StopOnRoundEndWrapper, RemoveNoopWrapper, SkipRedundantFramesWrapper

# Enable autoreload for all modules
# %load_ext autoreload
# %autoreload 2

# Training Configuration
CONFIG = {
    # Environment Settings
    "env_name": "ALE/Surround-v5",
    "game": "Surround",
    "mode": 0,
    "difficulty": 3,
    "frameskip": 1,
    
    # Directory Settings
    "base_dir": "./experiments",
    "video_subdir": "videos",
    "checkpoint_subdir": "checkpoints",
    "log_subdir": "logs",
    
    # Training Settings
    "num_episodes": 1000000,
    "render_every_n": 250,
    "save_every_n": 1000,
    
    # Environment Preprocessing
    "screen_size": 84,
    "grayscale": True,
    "frame_stack": 0,  # Set to 0 to disable
    "normalize": False, # Don't use this, normalization built into AtariPreprocessing wrapper
    "base_survival_reward": 0.01,  # Lower base reward since it will scale up
    # "survival_scaling_factor": 1.05, # Exponential growth factor
    "reward_scaling_factor": 1, # Scaling factor for rewards
    "stop_on_round_end": True,

    # Agent Selection
    # "agent_type": "reinforce",  # Options: "reinforce" or "dqn"
    "agent_type": "rainbow_dqn",
    
    # Shared Agent Settings
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "learning_rate": 1e-4, 
    "weight_decay": 0,
    "gamma": 0.99, # Discount factor
    "use_cnn": True,
    
    # REINFORCE-specific settings
    "reinforce": {
        "max_grad_norm": 0.5,
        "entropy_coef": 1e-4,
        # Network architecture
        "cnn_channels": [32, 64, 64],
        "cnn_kernel_sizes": [3, 3, 3],
        "cnn_strides": [2, 2, 1],
        "hidden_dims": [512]
    },
    
    # DQN-specific settings
    "dqn": {
        # Network architecture
        "cnn_channels": [32, 64, 64],
        "cnn_kernel_sizes": [8, 4, 3],
        "cnn_strides": [4, 2, 1],
        "hidden_dims": [64, 32],
        
        # Buffer settings
        "buffer_size": 100000,
        "batch_size": 64,
        "target_update_freq": 1, # Update target network every n steps, set to 1 (or something low) for soft update
        "tau": 0.001, # Soft update parameter, if 1 then hard update
        "update_freq": 1,
        "gradient_clip": 1.0,
        
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay": 0.99,
        "double_dqn": True,
        
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_end": 1.0,
        "per_beta_steps": 100000
    },

    # DQN-specific settings
    "dueling_dqn": {
        "cnn_channels": [6, 6, 6],
        "cnn_kernel_sizes": [8, 4, 3],
        "cnn_strides": [4, 2, 1],
        "hidden_dims": [512],
        
        # Buffer settings
        "buffer_size": 50000,
        "batch_size": 2048,
        "target_update_freq": 2, # Update target network every n steps, set to 1 (or something low) for soft update
        "tau": 0.001, # Soft update parameter, if 1 then hard update
        "update_freq": 1,
        "gradient_clip": 1.0,
        
        "eps_start": 1.0,
        "eps_end": 0.05,
        "eps_decay": 0.9995,
        "double_dqn": True,
        
        "per_alpha": 0.4,
        "per_beta_start": 0.4,
        "per_beta_end": 1.0,
        "per_beta_steps": 200000 
    },

    "rainbow_dqn": {
        "cnn_channels": [32, 64, 64],
        "cnn_kernel_sizes": [8, 4, 3],
        "cnn_strides": [4, 2, 1],
        "hidden_dims": [512],
        "buffer_size": 100000,
        "batch_size": 32,
        "target_update_freq": 4,
        "tau": 0.002,
        "update_freq": 4,
        "gradient_clip": 1.0,
        "double_dqn": True,
        "per_alpha": 0.6,
        "per_beta_start": 0.4,
        "per_beta_end": 1.0,
        "per_beta_steps": 100000,
        "n_steps": 3,
        "v_min": -10.0,
        "v_max": 10.0,
        "n_atoms": 51
    },
}


# In[2]:


def create_env(config, experiment_dir):
    """Create and wrap the environment according to configuration."""
    env = gym.make(
        config["env_name"],
        render_mode="rgb_array",
        mode=config["mode"],
        difficulty=config["difficulty"],
        frameskip=config["frameskip"]
    )

    env = SkipRedundantFramesWrapper(env)
    
    # Naive Stop on round end wrapper to detect end via reward (1 or -1)
    if config["stop_on_round_end"]:
        env = StopOnRoundEndWrapper(env, stop_reward=1)

    # Progressive reward wrapper
    # env = ProgressiveRewardWrapper(
    #     env,
    #     base_survival_reward=config["base_survival_reward"]
    # )
    env = TransformReward(env, lambda r: r + config["base_survival_reward"])
        
    if config["reward_scaling_factor"] != 1:
        env = TransformReward(env, lambda r: r * config["reward_scaling_factor"])
    
    # Preprocessing wrappers
    env = AtariPreprocessing(
        env,
        noop_max=4,
        frame_skip=1,
        screen_size=config["screen_size"],
        scale_obs=True,
        grayscale_obs=config["grayscale"],
        grayscale_newaxis=config["grayscale"] # add channel dim
    )

    # Video wrapper
    env = RecordVideo(
        env,
        video_folder=str(experiment_dir / config["video_subdir"]),
        episode_trigger=lambda episode_id: episode_id % config["render_every_n"] == 0,
        fps=60
    )
    
    env = ChannelFirstWrapper(env)
    
    if config["frame_stack"] > 0:
        env = FrameStackObservation(env, config["frame_stack"])
        env = ChannelWiseFrameStack(env)
    
    if config["normalize"]:
        env = NormalizeWrapper(env)
    
    env = RemoveNoopWrapper(env)

    return env

def create_agent_from_config(config: dict, env: gym.Env):
    """
    Creates an agent instance from a configuration dictionary by automatically
    mapping config parameters to constructor arguments.
    
    The function combines common parameters with agent-specific ones and uses
    dictionary unpacking for clean initialization.
    """
    # Common parameters shared across all agents
    common_params = {
        "env": env,
        "device": config.get("device", "cpu"),
        "use_cnn": config.get("use_cnn", False),
        "lr": config.get("learning_rate", 1e-3),
        "weight_decay": config.get("weight_decay", 0),
        "gamma": config.get("gamma", 0.99)
    }
    
    agent_type = config["agent_type"].lower()
    
    # Get the appropriate agent class
    agent_classes = {
        "reinforce": ReinforcePolicyGradientsAgent,
        "dqn": DQNAgent,
        "dueling_dqn": DuelingDQNAgent,
        "rainbow_dqn": RainbowDQNAgent
    }
    
    if agent_type not in agent_classes:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent_class = agent_classes[agent_type]
    
    # Combine common parameters with agent-specific ones
    agent_params = {**common_params}
    if agent_type in config:
        agent_params.update(config[agent_type])
    
    # Create and return the agent instance using dictionary unpacking
    return agent_class(**agent_params)


# In[3]:


def plot_training_history(history, save_dir):
    """Plot and save training metrics."""
    df = pd.DataFrame(history)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Define metrics groupings
    value_metrics = ['total_return', 'mean_q_value']  # DQN and REINFORCE
    loss_metrics = ['q_loss', 'policy_loss', 'entropy_loss', 'total_loss']  # Combined
    auxiliary_metrics = ['eps']  # DQN-specific
    
    # Filter to only metrics present in the data
    value_metrics = [m for m in value_metrics if m in df.columns]
    loss_metrics = [m for m in loss_metrics if m in df.columns]
    auxiliary_metrics = [m for m in auxiliary_metrics if m in df.columns]
    
    # Determine number of subplot rows needed
    n_rows = (bool(value_metrics) + bool(loss_metrics) + bool(auxiliary_metrics))
    
    # Create combined figure
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 5*n_rows))
    if n_rows == 1:
        axes = [axes]
    
    current_ax = 0
    
    # Plot value metrics
    if value_metrics:
        ax = axes[current_ax]
        for metric in value_metrics:
            ax.plot(df[metric], label=metric)
        ax.set_title('Value Metrics')
        ax.set_xlabel('Episode')
        ax.legend()
        current_ax += 1
    
    # Plot loss metrics
    if loss_metrics:
        ax = axes[current_ax]
        for metric in loss_metrics:
            ax.plot(df[metric], label=metric)
        ax.set_title('Loss Metrics')
        ax.set_xlabel('Episode')
        ax.legend()
        current_ax += 1
    
    # Plot auxiliary metrics
    if auxiliary_metrics:
        ax = axes[current_ax]
        for metric in auxiliary_metrics:
            ax.plot(df[metric], label=metric)
        ax.set_title('Auxiliary Metrics')
        ax.set_xlabel('Episode')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(save_dir / "training_history.png")
    plt.close()
    
    # Also save individual plots
    all_metrics = df.columns.tolist()
    excluded_metrics = {'total_steps', 'buffer_size'}
    
    for metric in all_metrics:
        if metric not in excluded_metrics:
            plt.figure(figsize=(10, 6))
            plt.plot(df[metric])
            plt.title(metric)
            plt.xlabel('Episode')
            plt.ylabel(metric)
            plt.savefig(save_dir / f"{metric}_history.png")
            plt.close()


# In[4]:


def save_checkpoint(agent, episode, history, save_dir, config):
    """Save agent checkpoint and training history."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create checkpoint directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save agent state
    agent_path = save_dir / f"agent_{config['agent_type']}_episode_{episode}_{timestamp}.pth"
    agent.save(agent_path)
    
    # Save training state
    checkpoint = {
        'episode': episode,
        'history': history,
        'config': config
    }
    checkpoint_path = save_dir / f"training_state_episode_{episode}_{timestamp}.pt"
    torch.save(checkpoint, checkpoint_path)
    
    return agent_path, checkpoint_path

def load_checkpoint(agent_path, config_path):
    """
    Load agent checkpoint and config.
    
    Args:
        agent_path: Path to .pth agent checkpoint
        config_path: Path to experiment config.json
    
    Returns:
        agent: Loaded agent instance
        config: Loaded config dict
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    # Create environment
    env = create_env(config, Path(config['base_dir']))
    
    # Create agent using helper function
    agent = create_agent_from_config(config, env)
    
    # Load agent state
    agent.load(agent_path)
    
    return agent, config


# In[5]:


def main(config):
    # Create experiment directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['game']}_{config['agent_type']}_{timestamp}"
    experiment_dir = Path(config["base_dir"]) / experiment_name
    
    # Create subdirectories
    for subdir in [config["video_subdir"], config["checkpoint_subdir"], config["log_subdir"]]:
        (experiment_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(experiment_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)
    
    # Create environment and agent
    env = create_env(config, experiment_dir=experiment_dir)
    agent = create_agent_from_config(config, env)
    
    print(f"\nAgent Type: {config['agent_type'].upper()}")
    print("Model Architecture:")
    print(agent.model if hasattr(agent, 'model') else agent.q_network)
    print("\nObservation Space:", env.observation_space)
    
    # Reset the environment and get the initial observation
    observation, _ = env.reset()

    # Check if the observation is channel-first (C, H, W) and permute if necessary
    if observation.shape[0] == 3 or observation.shape[0] == 1:
        observation = np.transpose(observation, (1, 2, 0))  # Convert to (H, W, C)

    # Plot and save the observation as an image to preview and view preprocessing
    plt.imshow(observation)
    plt.axis('off')  # Turn off axis
    image_path = experiment_dir / "model_obs_example.png"
    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()

    # Training loop
    history = []
    
    for episode in range(config["num_episodes"]):
        # Run episode
        results = agent.run_episode(env)
        history.append(results)
        
        # Print progress
        print(f"Episode {episode + 1}/{config['num_episodes']}: {results}")
        
        # Save checkpoint
        if (episode + 1) % config["save_every_n"] == 0:
            agent_path, checkpoint_path = save_checkpoint(
                agent,
                episode + 1,
                history,
                experiment_dir / config["checkpoint_subdir"],
                config
            )
            print(f"Saved checkpoint: {checkpoint_path}")

        if (episode + 1) % config["render_every_n"] == 0:
            # Plot training history
            plot_training_history(history, experiment_dir / config["log_subdir"])
    
    # Final save
    agent_path, checkpoint_path = save_checkpoint(
        agent,
        config["num_episodes"],
        history,
        experiment_dir / config["checkpoint_subdir"],
        config
    )
    
    # Plot training history
    plot_training_history(history, experiment_dir / config["log_subdir"])
    
    return agent, history, env, experiment_dir


# In[6]:


agent, history, env, exp_dir = main(CONFIG)


# In[19]:


# close the environment
env.close()

