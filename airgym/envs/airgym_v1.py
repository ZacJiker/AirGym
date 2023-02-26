import gym
import numpy as np

from typing import Tuple
from ..x_plane_connect import XPlaneConnect

class NotXPlaneRunning(Exception):
    pass

class AirGym(gym.Env):
    """AirGym environment for RL agents."""

    metadata = {"render.modes": ["human"]}

    def __init__(self) -> None:
        super().__init__()
        # Set action space to 4 dimensions (thrust, roll, pitch, yaw)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        # Set observation space to 6 dimensions (phi, theta, vz)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        # Store the X-Plane connection
        self.xp = XPlaneConnect()
        # Initiate X-Plane
        try:
            self.xp.getDREF("sim/test/test_float")
        except:
            raise NotXPlaneRunning("Please start X-Plane before running this script.")

    def _get_obs(self) -> Tuple[float, float, float, float]:
        # Return phi, theta, psi and vz
        vz = self.xp.getDREF("sim/flightmodel/position/local_vz")[0]
        theta = self.xp.getDREF("sim/flightmodel/position/theta")[0]
        phi = self.xp.getDREF("sim/flightmodel/position/phi")[0]        
        return np.array([vz, theta, phi])

    def reset(self) -> Tuple[float, float, float, float]:
        """Reset the environment to the initial state."""
        # Initiate time UTC to 11:00 AM
        self.xp.sendDREF("sim/time/zulu_time_sec", 19 * 3600)
        # Initiate position to KSEA (Seattle, WA)
        self.xp.sendPOSI([47.45, -122.30899, 2500, 0,    0,   0,  1])
        # Set angle of attack, velocity, and orientation
        data = [[18,   0, -998,   0, -998, -998, -998, -998, -998],
                [ 3, 130,  130, 130,  130, -998, -998, -998, -998],
                [16,   0,    0,   0, -998, -998, -998, -998, -998]]
        self.xp.sendDATA(data)
        # Return initial observation
        return self._get_obs()
    
    def step(self, action, ki: float = 0.1, ) -> Tuple[Tuple[float, float, float, float], float, bool, dict]:
        """Take a step in the environment.
        
        Args:
            action (Tuple[float, float, float, float]): The action to take."""
        # Set the action to the aircraft
        self.xp.sendCTRL(action)
        # Compute the reward
        obs = self._get_obs()
        reward = 10 - ((abs(obs[0]) * 0.1)**3 + (abs(obs[1]) * 0.01)**2 + (abs(obs[2]) * 0.001) + 0.001)
        # Send reward to X-Plane
        self.xp.sendTEXT("Reward: " + str(reward))
        # Return the next observation, reward, done and info
        return obs, reward, False, {}