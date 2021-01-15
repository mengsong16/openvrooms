'''
Quick example of usage of the run_experiment API.
'''
from all.experiments import plot_returns_100

import os
wd = os.getcwd()

import sys
sys.path.insert(0, wd)

from presets import dqn
from presets.models import *
from run_experiment import run_experiment

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
	parser.add_argument('--device',
						'-d',
						choices=['cuda:0', 'cuda:1', 'cpu'],
						default='cuda:0',
						help='cpu or gpu (default: cuda:0)')
	args = parser.parse_args()

	config_file = os.path.join(config_path,'turtlebot_relocate.yaml')
	config = parse_config(config_file)

	training_timesteps = config.get('training_timesteps')


	env = OpenRelocateEnvironment(gym_id="openrelocate-v0", 
		config_file=config_file, 
		mode=args.mode, 
		action_timestep=config.get('action_timestep'), 
		physics_timestep=config.get('physics_timestep'),
		device=torch.device(args.device),
		device_idx=0)

	agent = dqn(
		# Common settings
		device=args.device,
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
		agents=[agent],
		envs=[env],
		frames=training_timesteps,
		test_episodes=0
	)

	print("Done!")


if __name__ == "__main__":
	main()
