from gym.envs.registration import register

register(
    id="AirGym-v1",
    entry_point="airgym.envs:AirGym",
)