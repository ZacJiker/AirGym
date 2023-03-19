import gym
import sys

import numpy as np

from time import sleep
from loguru import logger

from scipy.spatial.distance import pdist

from airgym import XPlaneConnect
from airgym import action_space, observation_space

class NotXPlaneRunning(Exception):
    pass


class AirGym(gym.Env):
    """AirGym environment for RL agents."""

    metadata = {"render.modes": ["human"]}

    def __init__(self, address_ip: str = "0.0.0.0", port: int = 49009, timeout: int = 3600):
        """Initialize the environment.

        Raises:
            NotXPlaneRunning: If X-Plane is not running."""
        super().__init__()
        # Set action space to 4 dimensions (thrust, roll, pitch, yaw)
        self.action_space = action_space()
        # Set observation space to 6 dimensions (phi, theta, vz)
        self.observation_space = observation_space()
        # Store the X-Plane connection
        self.xp = XPlaneConnect(address_ip, port, 0, timeout)
        # Initiate X-Plane
        try:
            self.xp.getDREF("sim/test/test_float")
        except:
            logger.error("X-Plane is not running.")
            sys.exit(1)
            

    def _get_obs(self):
        """Get the observation from X-Plane.

        Returns:
            np.ndarray: The observation."""
        # Get the observation from X-Plane
        raw_data = self.xp.getDREFs([
            "sim/flightmodel/position/phi",
            "sim/flightmodel/position/theta",
            "sim/flightmodel/position/psi",
            "sim/flightmodel/position/local_vx",
            "sim/flightmodel/position/local_vy",
            "sim/flightmodel/position/local_vz",
            "sim/flightmodel/position/P",
            "sim/flightmodel/position/Q",
            "sim/flightmodel/position/R",]
        )
        return np.array([item[0] for item in raw_data], dtype=np.float64)

    def _compute_reward(self, obs: np.ndarray, target: np.ndarray, sigma: float = 0.45):
        """Compute the reward.

        Args:
            obs (np.ndarray): The observation.
            target (np.ndarray): The target observation.
            sigma (float, optional): The sigma parameter. Defaults to 0.1.

        Returns:
            float: The reward."""
        # Calculate the distance between the observation and the target
        distance = pdist(np.array([obs, target]), 'cosine')
        # Calculate the reward
        return np.exp(-distance ** 2 / sigma ** 2)[0]

    def reset(self):
        """Reset the environment to the initial state.

        Returns:
            np.ndarray: The initial observation."""
        # Initiate local time to 12:00
        self.xp.sendDREF("sim/time/local_time_sec", 43200)
        # Set the aircraft to LFMT (Montpellier, France)
        self.xp.sendDREF("sim/flightmodel/position/latitude", 43.576)
        self.xp.sendDREF("sim/flightmodel/position/longitude", 3.963)
        # Set the aircraft to the initial position
        self.xp.sendDREF("sim/flightmodel/position/local_x", 0)
        self.xp.sendDREF("sim/flightmodel/position/local_y", 762)
        self.xp.sendDREF("sim/flightmodel/position/local_z", 0)
        # Set the aircraft to the initial orientation
        self.xp.sendDREF("sim/flightmodel/position/theta", 0)
        self.xp.sendDREF("sim/flightmodel/position/phi", 0)
        self.xp.sendDREF("sim/flightmodel/position/psi", 120)
        # Set the aircraft to the initial velocity
        self.xp.sendDREF("sim/flightmodel/position/local_vx", 60)
        self.xp.sendDREF("sim/flightmodel/position/local_vy", 0)
        self.xp.sendDREF("sim/flightmodel/position/local_vz", 0)
        # Set the aircraft to the initial angular velocity
        self.xp.sendDREF("sim/flightmodel/position/P", 0)
        self.xp.sendDREF("sim/flightmodel/position/Q", 0)
        self.xp.sendDREF("sim/flightmodel/position/R", 0)
        # Wait for the aircraft to be in the initial position
        sleep(0.0003)
        # Return initial observation
        return self._get_obs()

    def step(self, action):
        """Take a step in the environment.

        Args:
            action (Tuple[float, float, float, float]): The action to take.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: The next observation, reward, done, and info."""
        # Set the action to the aircraft
        self.xp.sendCTRL(action)
        # Add a delay to make sure the action is sent
        sleep(0.0003)
        # Get the next observation
        obs = self._get_obs()
        # Calculate the reward based on the observation and the target psi at 120°
        # and velocity at 60 m/s
        target_state = np.array([0, 0, 120, 60, 0, 0, 0, 0, 0], dtype=np.float64)
        # 
        if abs(obs[2] - target_state[2]) > 1.5 and abs(obs[3] - target_state[3]) < 1.5:
            reward = self._compute_reward(obs, target_state, sigma=0.85)
        else:
            reward = - self._compute_reward(obs, target_state)

        return obs, reward, False, {}
        

    def render(self, mode: str = "human"):
        """Render the environment.

        Args:
            mode (str, optional): The mode to render the environment. Defaults to "human"."""
        NotImplementedError()

    def close(self):
        """Close the environment."""
        self.xp.close()