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
