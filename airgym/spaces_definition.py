# AirGym: A Reinforcement Learning Environment 🚀, GPL-3.0 License

import gym
import numpy as np


def action_space():
    """Get the action space.

    Returns:
        gym.spaces.Box: The action space.
    """
    return gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float64)


def observation_space():
    """Get the observation space.

    Returns:
        gym.spaces.Box: The observation space.
    """
    return gym.spaces.Box(
        low=np.array([-180, -90, -180, -100, -100, -100, -200, -200, -200]),
        high=np.array([180, 90, 180, 100, 100, 100, 200, 200, 200]),
        shape=(9,),
        dtype=np.float64
    )
