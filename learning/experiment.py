'''
Quick example of usage of the run_experiment API.
'''
from all.experiments import run_experiment, plot_returns_100

import os
wd = os.getcwd()

import sys
sys.path.insert(0, wd)

from presets import dqn
from presets.models import *
from openvrooms.envs.relocate_env_wrapper import OpenRelocateEnvironment

from openvrooms.config import *

import argparse
from gibson2.utils.utils import parse_config
import torch

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--mode',
						'-m',
						choices=['headless', 'gui', 'iggui'],
						default='headless',
						help='which mode for simulation (default: headless)')
	args = parser.parse_args()

	config_file = os.path.join(config_path,'turtlebot_relocate.yaml')
	config = parse_config(config_file)

	training_timesteps = config.get('training_timesteps')

	env = OpenRelocateEnvironment(gym_id="openrelocate-v0", 
		config_file=config_file, 
		mode=args.mode, 
		action_timestep=config.get('action_timestep'), 
		physics_timestep=config.get('physics_timestep'),
		device=torch.device('cuda:0'),
		device_idx=0)

	alg = dqn(
		# Common settings
		device='cuda:0',
		discount_factor=config.get('discount_factor'),
		# Adam optimizer settings
		lr=config.get('learning_rate'),
		# Training settings
		minibatch_size=config.get('minibatch_size'),
		update_frequency=config.get('update_frequency'),
		target_update_frequency=config.get('target_update_frequency'),
		# Replay buffer settings
		replay_start_size=config.get('replay_start_size'),
		replay_buffer_size=config.get('replay_buffer_size'),
		# Exploration settings
		initial_exploration=config.get('initial_exploration'),
		final_exploration=config.get('final_exploration'),
		final_exploration_frame=config.get('final_exploration_frame'),
		# Model construction
		model_constructor=fc_relu_q)


	run_experiment(
		[alg],
		[env],
		training_timesteps,
	)
	plot_returns_100('runs', timesteps=training_timesteps)


if __name__ == "__main__":
	main()
