import gym
import wandb

import airgym

import numpy as np

from gym.wrappers.time_limit import TimeLimit

from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from wandb.integration.sb3 import WandbCallback

config = {
    "policy_type": "MlpPolicy",
    "total_timesteps": 2000000,
    "env_name": "AirGym-v1",
    "address_ip": "192.168.1.175",
    "time_limit": 1000
}

run = wandb.init(
    project="airgym",
    name="sac_2M",
    config=config,
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True
)

if __name__ == '__main__':
    # Create environment
    env = gym.make(config["env_name"], address_ip=config["address_ip"])
    env = TimeLimit(env, max_episode_steps=config["time_limit"])
    # Wrap environment with Monitor and DummyVecEnv
    env = Monitor(env, allow_early_resets=True)
    #env = DummyVecEnv([lambda: env])
    # Train model
    model = SAC(config["policy_type"], env, verbose=1,
                tensorboard_log=f"runs/{run.id}")
    model.learn(total_timesteps=config["total_timesteps"], callback=WandbCallback(
        gradient_save_freq=100, verbose=2, model_save_path=f"models/{run.id}"))

    run.finish()
