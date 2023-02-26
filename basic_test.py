import gym
import airgym

from gym.wrappers.time_limit import TimeLimit

env = gym.make('AirGym-v1')
env = TimeLimit(env, max_episode_steps=1000)

obs = env.reset()

while True:
    obs, reward, done, info = env.step(env.action_space.sample())
    if done:
        break