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
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(6,))
        # Store the X-Plane connection
        self.xp = XPlaneConnect()
        # Initiate X-Plane
        try:
            self.xp.getDREF("sim/flightmodel/position/latitude")
        except:
            raise Exception("X-Plane is not running.")

    def _get_obs(self) -> Tuple[float, float, float, float, float, float]:
        pass 

    def reset(self) -> Tuple[float, float, float, float, float, float]:
        # Initiate time UTC to 11:00 AM
        self.xp.sendDREF("sim/time/zulu_time_sec", 19 * 3600)
        # Initiate location to KSEA (Seattle, WA)
        self.xp.sendDREF("sim/flightmodel/position/latitude", 47.449)
        self.xp.sendDREF("sim/flightmodel/position/longitude", -122.309)
        # Initiate state at 2500 ft, 120 knots, 0 degrees pitch, 0 degrees roll
        self.xp.sendDREF("sim/flightmodel/position/elevation", 2500)
        self.xp.sendDREF("sim/flightmodel/position/true_airspeed", 120)
        self.xp.sendDREF("sim/flightmodel/position/theta", 0)
        self.xp.sendDREF("sim/flightmodel/position/phi", 0)
        # Return initial observation
        return self._get_obs()