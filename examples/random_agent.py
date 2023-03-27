import gym
import airgym

from gym.wrappers.time_limit import TimeLimit

from time import sleep

env = gym.make('AirGym-v1', address_ip='192.168.1.175')
env = TimeLimit(env, max_episode_steps=300)

episod = 0 

while episod < 10:
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        print('episod : ', episod, 'reward : ', reward, 'done : ', done, 'info : ', info)
        sleep(0.0003)
    episod += 1

env.close()