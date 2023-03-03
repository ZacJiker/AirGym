import gym
import time
import numpy as np

from typing import Tuple
from ..x_plane_connect import XPlaneConnect

class NotXPlaneRunning(Exception):
    pass

class AirGym(gym.Env):
    """AirGym environment for RL agents."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, xp: XPlaneConnect):
        """Initialize the environment.
        
        Raises:
            NotXPlaneRunning: If X-Plane is not running."""
        super().__init__()
        # Set action space to 4 dimensions (thrust, roll, pitch, yaw)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        # Set observation space to 6 dimensions (phi, theta, vz)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        # Store the X-Plane connection
        self.xp = xp
        # Initiate X-Plane
        try:
            self.xp.getDREF("sim/test/test_float")
        except:
            raise NotXPlaneRunning("Please start X-Plane before running this script.")

    def _get_obs(self):
        """Get the observation from X-Plane.
        
        Returns:
            np.ndarray: The observation."""
        # # Set the connection to the X-Plane server
        # self.xp.setCONN(port=49009)
        # Get the observation from X-Plane
        obs = self.xp.getDREFs([
            "sim/flightmodel/position/local_vz", 
            "sim/flightmodel/position/theta", 
            "sim/flightmodel/position/phi"]
            )      
        return obs

    def reset(self):
        """Reset the environment to the initial state.
        
        Returns:
            np.ndarray: The initial observation."""
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
    
    def step(self, action):
        """Take a step in the environment.
        
        Args:
            action (Tuple[float, float, float, float]): The action to take.
            
        Returns:
            Tuple[np.ndarray, float, bool, dict]: The next observation, reward, done and info."""
        # Set the action to the aircraft
        #self.xp.sendCTRL(action)
        # Get the next observation
        obs = self._get_obs()
        # Calculate the reward
        reward = 10 - (sum(obs) ** 0.5)
        # Send reward to X-Plane
        self.xp.sendTEXT("Reward: " + str(round(reward, 3)))
        # Add a delay to make sure the reward is sent
        time.sleep(0.1)
        # Return the next observation, reward, done and info
        return obs, reward, False, {}