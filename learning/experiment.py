'''
Quick example of usage of the run_experiment API.
'''
from all.experiments import run_experiment, plot_returns_100
#from all.presets.classic_control import dqn
import os
wd = os.getcwd()
#wd = os.path.join(wd, "presets")

import sys
sys.path.insert(0, wd)

from presets import dqn
from presets.models import *
#from all.environments import GymEnvironment
from openvrooms.envs.relocate_env_wrapper import OpenRelocateEnvironment

from openvrooms.config import *

def main():
    timesteps = 40000

    env = OpenRelocateEnvironment(name="openrelocate-v0", 
    	config_file=os.path.join(config_path,'turtlebot_relocate.yaml'), 
    	mode="headless", 
    	action_timestep=1.0 / 10.0, 
    	physics_timestep=1.0 / 40.0)

    alg = dqn(
        # Common settings
        device="cpu",
        discount_factor=0.99,
        # Adam optimizer settings
        lr=1e-3,
        # Training settings
        minibatch_size=64,
        update_frequency=1,
        target_update_frequency=100,
        # Replay buffer settings
        replay_start_size=1000,
        replay_buffer_size=10000,
        # Exploration settings
        initial_exploration=1.,
        final_exploration=0.,
        final_exploration_frame=10000,
        # Model construction
        model_constructor=fc_relu_q)

    run_experiment(
        [alg],
        [env],
        timesteps,
    )
    plot_returns_100('runs', timesteps=timesteps)

if __name__ == "__main__":
    main()
