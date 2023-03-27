# AirGym: A Reinforcement Learning Environment 🚀, GPL-3.0 License

import gym

import numpy as np

from time import sleep

from scipy.spatial.distance import pdist

from airgym import XPlaneConnect
from airgym import action_space, observation_space


class NotXPlaneRunning(Exception):
    pass


class AirGym(gym.Env):
    """The AirGym environment.

    Attributes:
        action_space (gym.spaces.Box): The action space.
        observation_space (gym.spaces.Box): The observation space.
        xp (XPlaneConnect): The X-Plane connection.
    """

    metadata = {"render.modes": ["human"]}

    def __init__(self, address_ip: str = "0.0.0.0", port: int = 49009, timeout: int = 3600):
        """Initialize the environment.

        Args:
            address_ip (str, optional): The IP address of the X-Plane computer. Defaults to localhost.
            port (int, optional): The port of the X-Plane computer. Defaults to 49009.
            timeout (int, optional): The timeout of the X-Plane connection. Defaults to 3600.

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
            raise NotXPlaneRunning("X-Plane is not running.")

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
            sigma (float, optional): The sigma parameter. Defaults to 0.45.

        Returns:
            float: The reward."""
        # Calculate the distance between the observation and the target
        distance = pdist(np.array([obs, target]), 'cosine')
        # Calculate the reward
        return np.exp(-distance ** 2 / sigma ** 2)[0]

    def reset(self):
        """Reset the environment to the initial state.

        Returns:
            np.ndarray: The initial obs
        """
        drefs = [
            "sim/time/local_time_sec",
            "sim/flightmodel/position/latitude",
            "sim/flightmodel/position/longitude",
            "sim/flightmodel/position/local_x",
            "sim/flightmodel/position/local_y",
            "sim/flightmodel/position/local_z",
            "sim/flightmodel/position/phi",
            "sim/flightmodel/position/theta",
            "sim/flightmodel/position/psi",
            "sim/flightmodel/position/local_vx",
            "sim/flightmodel/position/local_vy",
            "sim/flightmodel/position/local_vz",
            "sim/flightmodel/position/P",
            "sim/flightmodel/position/Q",
            "sim/flightmodel/position/R",
        ]
        self.xp.sendDREFs(drefs=drefs, values=[
                          43200, 43.576, 3.963, 0, 5000, 0, 0, 0, 60, 0, 0, 0, 0, 0, 0])
        # Wait for the aircraft to be in the initial position
        sleep(0.01)
        # Return initial observation
        return self._get_obs()

    def step(self, action):
        """Take a step in the environment.
        
        Args:
            action (np.ndarray): The action to take.
            
        Returns:
            np.ndarray: The observation.
            float: The reward.
            bool: If the episode is done.
            dict: The info.
        """
        # Set the action to the aircraft
        self.xp.sendCTRL(action)
        # Add a delay to make sure the action is sent
        sleep(0.01)
        # Get the next observation
        try:
            obs = self._get_obs()
        except:
            # If the aircraft is out of the simulation, reset the environment
            obs = self.reset()
        # Calculate the reward based on the observation and the target psi at 120°
        # and velocity_x at 60 m/s
        target_state = np.array(
            [0, 0, 120, 60, 0, 0, 0, 0, 0], dtype=np.float64)
        # If the aircraft has a psi between 119° and 121° and a velocity between 59 m/s and 61 m/s
        # then the reward is high, otherwise penalize the agent
        if np.sum(abs(obs - target_state)) < (obs.shape[0] * 1.5):
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
