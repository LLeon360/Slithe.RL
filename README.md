# Atari Surround RL

This repository implements reinforcement learning (RL) approaches to play the Atari game Surround. The goal is to train agents that can effectively learn and play the game using various RL algorithms.

## Repository Structure
```
📦 
├─ Atari.ipynb
├─ LunarLander.ipynb
├─ agents
│  ├─ agent.py
│  └─ reinforce.py
├─ models
│  ├─ cnns.py
│  └─ mlps.py
├─ setup.sh
└─ wrappers
   └─ gym_wrappers.py
```

- **agents/**: Contains the implementation of RL algs, including REINFORCE.
- **models/**: Defines neural network architectures used by the agents.
- **wrappers/**: Custom Gym wrappers for preprocessing and reward shaping.
- **Atari.ipynb**: Jupyter Notebook for training and evaluating agents on Atari Surround.
- **LunarLander.ipynb**: Jupyter Notebook for training and evaluating agents on Lunar Lander.
- **setup.sh**: Script to set up the required environment and dependencies.

video and checkpoint directory will also be created automatically to record episodes and model weights.

## Quick Start 

**Set up the environment:**
```sh
./setup.sh
```

If setup doesn't work due to not automatically activating the conda env, activate it with `conda activate atari` or whatever your env is called and then copy the remaining install commands.

