from gym.envs.registration import register

from airgym.spaces_definition import action_space, observation_space
from airgym.x_plane_connect import XPlaneConnect

register(
    id="AirGym-v1",
    entry_point="airgym.envs:AirGym",
)