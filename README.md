![GPLv3 License](https://img.shields.io/badge/License-GPL%20v3-yellow.svg)
![GitHub Workflow Status (with branch)](https://img.shields.io/github/actions/workflow/status/zacjiker/airgym/build-and-publish.yml)
![GitHub all releases](https://img.shields.io/github/downloads/zacjiker/airgym/total)
![PyPI](https://img.shields.io/pypi/v/airgym)

# XPlane Gym Environment

This project provides an OpenAI Gym environment for training reinforcement learning agents on an XPlane simulator. The environment allows agents to control an aircraft and receive rewards based on how well they perform a task, such as flying a certain trajectory or landing safely.

## Installation

To install the package, run the following command:

```bash
  pip install airgym
```
    
## Usage/Examples

To use the environment in your Python code, you can import it as follows:

```python
import airgym
import gym

# If XPlane is running on the same machine, you can use the default address and port. 
# Or, set ip address and port according to your configuration.
env = gym.make('AirGym-v1')

episods = 0

for episod in range(episods):
    obs = env.reset()
    done = False

    while not done:
        actions = env.action_space.sample()
        obs, reward, done, info = env.step(action)

env.close()
```
