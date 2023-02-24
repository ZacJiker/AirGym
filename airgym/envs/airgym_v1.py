import gym

from typing import Tuple
from ..x_plane_connect import XPlaneConnect

class AirGym(gym.Env):
    """AirGym environment for RL agents."""

    metadata = {"render.modes": ["human"]}

    def __init__(self) -> None:
        super().__init__()
        # Set action space to 4 dimensions (thrust, roll, pitch, yaw)
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        # Set observation space to 6 dimensions (phi, theta, psi, vx, vy, vz)
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        # Store the X-Plane connection
        self.xp = XPlaneConnect()
        # Initiate X-Plane
        try:
            self.xp.getDREF("sim/flightmodel/position/latitude")
        except:
            raise Exception("X-Plane is not running.")

    def _get_obs(self) -> Tuple[float, float, float, float]:
        # Return PHI, THETA, PSI, Vz, Vx, Vy
        phi = self.xp.getDREF("sim/flightmodel/position/phi")[0]
        theta = self.xp.getDREF("sim/flightmodel/position/theta")[0]
        psi = self.xp.getDREF("sim/flightmodel/position/psi")[0]
        vz = self.xp.getDREF("sim/flightmodel/position/local_vz")[0]
        return [phi, theta, psi, vz]

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
    
    def step(self, action) -> Tuple[Tuple[float, float, float, float], float, bool, dict]:
        """Take a step in the environment.
        
        Args:
            action (Tuple[float, float, float, float]): The action to take."""
        # Set the action to the aircraft
        self.xp.sendCTRL(action)
        # Return the next observation, reward, and done
        return self._get_obs(), 0, False, {}