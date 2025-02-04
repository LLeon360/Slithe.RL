import gymnasium as gym
import numpy as np

# Define a wrapper for Gym env to add survival rewards
class RewardWrapper(gym.Wrapper):
    def __init__(self, env, survival_reward=0.0001):
        super(RewardWrapper, self).__init__(env)
        self.survival_reward = survival_reward

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Add a small positive reward for surviving
        reward += self.survival_reward
        return obs, reward, terminated, truncated, info

class ProgressiveRewardWrapper(gym.Wrapper):
    '''
    Offers a piecewise reward system based on the number of steps survived
    '''
    def __init__(self, env, base_survival_reward=0.0001):
        super(ProgressiveRewardWrapper, self).__init__(env)
        self.base_survival_reward = base_survival_reward
        self.total_steps = 0
        self.round_steps = 0
        self.rounds = 0

        # Define step thresholds and their corresponding rewards for survival
        self.reward_tiers = {
            120: self.base_survival_reward * 0.01, 
            140: self.base_survival_reward,      # Basic survival
            150: self.base_survival_reward * 1.5, # Mild
            180: self.base_survival_reward * 6, # Decent performance
            250: self.base_survival_reward * 10, # Good performance
            float('inf'): self.base_survival_reward * 12  # Excellent
        }
        
        # Store sorted thresholds once
        self.sorted_thresholds = sorted(self.reward_tiers.keys())

    def get_survival_reward(self, steps):
        """Get the appropriate reward based on current step count"""
        for threshold in self.sorted_thresholds:
            if steps < threshold:
                return self.reward_tiers[threshold]
        return self.reward_tiers[float('inf')]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        if abs(reward) == 1:
            self.round_steps = 0
            self.rounds += 1
            if reward == 1:
                reward = 100
                print("WON ROUND!!!!!!")
        else:
            survival_reward = self.get_survival_reward(self.round_steps)
            reward += survival_reward
            self.round_steps += 1
        
        self.total_steps += 1
        return obs, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self.round_steps = 0
        self.rounds = 0
        return self.env.reset(*args, **kwargs)
    
class ChannelFirstWrapper(gym.Wrapper):
    def __init__(self, env):
        super(ChannelFirstWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
			low=0,
			high=255,
			shape=(env.observation_space.shape[2], env.observation_space.shape[0], env.observation_space.shape[1]),
			dtype=env.observation_space.dtype
		)

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return obs.transpose(2, 0, 1), info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs.transpose(2, 0, 1), reward, terminated, truncated, info
    
class NormalizeWrapper(gym.Wrapper):
    def __init__(self, env):
        super(NormalizeWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
			low=0,
			high=1,
            shape=env.observation_space.shape,
            # convert to float
			dtype=np.float32
		)
    
    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)

        return obs / 255.0, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = obs / 255.0

        return obs, reward, terminated, truncated, info
    
class ChannelWiseFrameStack(gym.Wrapper):
    '''
    Given some framestacked observation passed through gym.wrapper.FrameStackObservation
    with [frames, channels, height, width] as the shape of the observation
    this wrapper will convert the observation to [frames * channels, height, width]
    '''
    def __init__(self, env):
        super(ChannelWiseFrameStack, self).__init__(env)
        
        self.frames = env.observation_space.shape[0]
        self.channels = env.observation_space.shape[1]
        self.height = env.observation_space.shape[2]
        self.width = env.observation_space.shape[3]
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(self.frames * self.channels, self.height, self.width),
            dtype=env.observation_space.dtype
        )

    def reset(self, *args, **kwargs):
        obs, info = self.env.reset(*args, **kwargs)
        return np.transpose(obs, (0, 2, 3, 1)).reshape(self.frames * self.channels, self.height, self.width), info
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return np.transpose(obs, (0, 2, 3, 1)).reshape(self.frames * self.channels, self.height, self.width), reward, terminated, truncated, info
