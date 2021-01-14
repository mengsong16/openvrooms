'''
Quick example of usage of the run_experiment API.
'''
from all.experiments import run_experiment, plot_returns_100
from all.presets.classic_control import dqn
from all.environments import GymEnvironment
from openvrooms.envs.relocate_env_wrapper import OpenRelocateEnvironment
import os
from openvrooms.config import *

def main():
    timesteps = 40000

    env = OpenRelocateEnvironment(name="openrelocate-v0", config_file=os.path.join(config_path,'turtlebot_relocate.yaml'), mode="headless", action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0)

    print(env.state_space.shape)
    '''
    run_experiment(
        [dqn()],
        [env],
        timesteps,
    )
    plot_returns_100('runs', timesteps=timesteps)
	'''
if __name__ == "__main__":
    main()
