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

from openvrooms.envs.openroom_env_wrapper import OpenRoomEnvironment

from openvrooms.config import *

import argparse
from gibson2.utils.utils import parse_config
import torch
from learning_utils import set_seed_everywhere

def deploy(agent_dir, env, record=False):
	agent = GreedyAgent.load(agent_dir, env)

	print('----------------- Deployment Start --------------------')
	action = None
	returns = 0

	if record:
		env.start_video_recorder()

	env.reset()
	episode = 0
	while episode < 3:
		# run policy
		# env.reward is just a place holder
		action = agent.act(env.state, env.reward)

		# step env
		if not env.done:
			env.step(action)
			returns += env.reward
		else:
			# next episode
			print('episode: %d, returns: %f'%(episode, returns))
			episode += 1

			env.reset()
			returns = 0
	
	env.close()	

	if record:
		env.save()
	print("-----------------Done!-----------------")	
	
def train(args, config, env, control_mode='state'):
	training_timesteps = config.get('training_timesteps')
	print('----------------- Training Start --------------------')
	if control_mode == 'state':
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
	else:	
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
			model_constructor=vision_q)
	


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
	parser.add_argument('--env',
						'-e',
						choices=['nav', 'rel'],
						default='nav',
						help='navigation or relocation')
	parser.add_argument("--record", help="record training or deploy process as video",
                    action="store_true") # need to ensure that rgb is included in output

	args = parser.parse_args()

	if args.env == "rel":
		config_file = os.path.join(config_path,'turtlebot_relocate.yaml')
	else:
		config_file = os.path.join(config_path,'turtlebot_navigate.yaml')
	
	config = parse_config(config_file)
	

	# if vision-based, need to stack frames
	output = config.get('output')
	
	if 'task_obs' not in output:
		frame_stack = config.get('frame_stack')
		control_mode = 'vision'
	else:
		frame_stack = None	
		control_mode = 'state'	
	
	
	if args.record:
		save_path = "./runs/dqn_empty_room"
		print("Will save recorded video at: %s"%(save_path))
	else:
		save_path = None	

	# fix random seeds before creating the environment
	use_seed = 1
	set_seed_everywhere(seed=use_seed)
	print('--------------------------------------------')
	print('Random seed: %d'%(use_seed))
	print('--------------------------------------------')

	if args.env == "rel":
		env = OpenRoomEnvironment(gym_id="openrelocate-v0", 
			config_file=config_file, 
			mode=args.render, 
			action_timestep=config.get('action_timestep'), 
			physics_timestep=config.get('physics_timestep'),
			device=torch.device(args.device),
			device_idx=0, frame_stack=frame_stack, save_format='mp4', save_path=save_path)
	else:	
		env = OpenRoomEnvironment(gym_id="opennavigate-v0", 
			config_file=config_file, 
			mode=args.render, 
			action_timestep=config.get('action_timestep'), 
			physics_timestep=config.get('physics_timestep'),
			device=torch.device(args.device),
			device_idx=0, frame_stack=frame_stack, save_format='mp4', save_path=save_path)

	if args.mode == "train":
		train(args, config, env, control_mode)
	else:
		agent_dir = "./runs/dqn_empty_room"
		deploy(agent_dir, env, record=args.record)

if __name__ == "__main__":
	run()
