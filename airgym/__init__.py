# AirGym: A Reinforcement Learning Environment 🚀, GPL-3.0 License

from gym.envs.registration import register

__version__ = "0.0.2"

register(
    id="AirGym-v1",
    entry_point="airgym.envs:AirGym",
)