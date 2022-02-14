import numpy as np
from gym.spaces import Box
from gym import ObservationWrapper
import torch
import cv2

class ResizeObservation(ObservationWrapper):
    r"""Downsample the image observation to a square image."""

    def __init__(self, env, shape):
        super(ResizeObservation, self).__init__(env)
        if isinstance(shape, int):
            shape = (shape, shape)
        assert all(x > 0 for x in shape), shape

        self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        return cv2.resize(observation.astype(np.float32), self.shape[::-1], interpolation=cv2.INTER_AREA)

class TensorObservation(ObservationWrapper):
    def __init__(self, env):
        super(TensorObservation, self).__init__(env)
        self.observation_space = Box(low=self.observation_space.low, high=self.observation_space.high, shape=self.observation_space.shape, dtype=np.float32)

    def observation(self, observation):
        return torch.as_tensor(observation, dtype=torch.float32)