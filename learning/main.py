from all.experiments import plot_returns_100
import time
import os
wd = os.getcwd()

import sys
sys.path.insert(0, wd)

from presets import dqn
from presets.models import *
from run_experiment import run_experiment
from greedy_agent import GreedyAgent

from openvrooms.envs.relocate_env_wrapper import OpenRelocateEnvironment

from openvrooms.config import *

import argparse
from gibson2.utils.utils import parse_config
import torch

def deploy(agent_dir, env, fps=60):
	agent = GreedyAgent.load(agent_dir, env)

	print('----------------- Deployment Start --------------------')
	action = None
	returns = 0
	env.reset()
	while True:
		# run policy
		# env.reward is just a place holder
		action = agent.act(env.state, env.reward)

		# step env
		if not env.done:
			env.step(action)
			returns += env.reward
		else:
			# next episode
			print('returns:', returns)
			env.reset()
			returns = 0
		
	print("-----------------Done!-----------------")	
	
def train(args, config, env):
	training_timesteps = config.get('training_timesteps')
	print('----------------- Training Start --------------------')
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

	print("-----------------Done!-----------------")

def run():
	parser = argparse.ArgumentParser()
	parser.add_argument('--render',
						'-r',
						choices=['headless', 'gui', 'iggui'],
						default='headless',
						help='which mode for simulation (default: headless)')
	parser.add_argument('--device',
						'-d',
						choices=['cuda:0', 'cuda:1', 'cpu'],
						default='cuda:0',
						help='cpu or gpu (default: cuda:0)')
	parser.add_argument('--mode',
						'-m',
						choices=['train', 'deploy'],
						default='train',
						help='train or deploy (train)')

	args = parser.parse_args()

	config_file = os.path.join(config_path,'turtlebot_relocate.yaml')
	config = parse_config(config_file)


	env = OpenRelocateEnvironment(gym_id="openrelocate-v0", 
		config_file=config_file, 
		mode=args.render, 
		action_timestep=config.get('action_timestep'), 
		physics_timestep=config.get('physics_timestep'),
		device=torch.device(args.device),
		device_idx=0)

	if args.mode == "train":
		train(args, config, env)
	else:
		agent_dir = "./runs/dqn"
		deploy(agent_dir, env)

if __name__ == "__main__":
	run()
