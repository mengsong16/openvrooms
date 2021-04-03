from gibson2.utils.utils import quatToXYZW
from gibson2.envs.env_base import BaseEnv
from gibson2.tasks.room_rearrangement_task import RoomRearrangementTask
from gibson2.tasks.point_nav_fixed_task import PointNavFixedTask
from gibson2.tasks.point_nav_random_task import PointNavRandomTask
from gibson2.tasks.interactive_nav_random_task import InteractiveNavRandomTask
from gibson2.tasks.dynamic_nav_random_task import DynamicNavRandomTask
from gibson2.tasks.reaching_random_task import ReachingRandomTask
from gibson2.sensors.scan_sensor import ScanSensor
from gibson2.sensors.vision_sensor import VisionSensor
from gibson2.robots.robot_base import BaseRobot
from gibson2.external.pybullet_tools.utils import stable_z_on_aabb
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.envs.env_base import BaseEnv

#from gibson2.robots.turtlebot_robot import Turtlebot
from openvrooms.robots.turtlebot import Turtlebot

from gibson2.robots.husky_robot import Husky
from gibson2.robots.ant_robot import Ant
from gibson2.robots.humanoid_robot import Humanoid
from gibson2.robots.jr2_robot import JR2
from gibson2.robots.jr2_kinova_robot import JR2_Kinova
from gibson2.robots.freight_robot import Freight
from gibson2.robots.fetch_robot import Fetch
from gibson2.robots.locobot_robot import Locobot


from openvrooms.tasks.navigate_goal_fixed_task import NavigateGoalFixedTask
from openvrooms.scenes.navigate_scene import NavigateScene
from openvrooms.sensors.external_vision_sensor import ExternalVisionSensor
from openvrooms.config import *
from openvrooms.envs.relocate_env import RelocateEnv

from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
import gym
import numpy as np
import pybullet as p
import time
import logging

#from gibson2.simulator import Simulator
from openvrooms.simulator.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

class NavigateEnv(RelocateEnv):
	"""
	iGibson Environment (OpenAI Gym interface)
	"""

	def __init__(
		self,
		config_file,
		scene_id=None,
		mode='headless',
		device_idx=0,
		render_to_tensor=False,
		automatic_reset=False,
	):
		"""
		:param config_file: config_file path
		:param scene_id: override scene_id in config file
		:param mode: headless, gui, iggui
		:param action_timestep: environment executes action per action_timestep second
		:param physics_timestep: physics timestep for pybullet
		:param device_idx: which GPU to run the simulation and rendering on
		:param render_to_tensor: whether to render directly to pytorch tensors
		:param automatic_reset: whether to automatic reset after an episode finishes
		"""
		
		super(NavigateEnv, self).__init__(config_file=config_file,
										 scene_id=scene_id,
										 mode = mode,
										 device_idx=device_idx,
										 render_to_tensor=render_to_tensor,
										 automatic_reset=automatic_reset)
		

	def load_scene_robot(self):
		"""
		Import the scene and robot (but have not reset their poses)
		"""
		if self.config['scene'] == 'navigate':
			scene_id = self.config['scene_id']
			n_obstacles = self.config.get('obs_num', 0)
			if "multi_band" in scene_id:
				scene = NavigateScene(scene_id=scene_id, n_obstacles=n_obstacles, multi_band=True)
			else:	
				scene = NavigateScene(scene_id=scene_id, n_obstacles=n_obstacles)
			
			self.simulator.import_scene(scene, load_texture=self.config.get('load_texture', True))
			self.scene = scene
		else:
			raise Exception(
				'unknown scene type: {}'.format(self.config['scene']))

		self.load_robot()



	def load_task_setup(self):
		"""
		Load task setup
		"""
		self.initial_pos_z_offset = self.config.get(
			'initial_pos_z_offset', 0.1)
		# s = 0.5 * G * (t ** 2)
		drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
		assert drop_distance < self.initial_pos_z_offset, \
			'initial_pos_z_offset is too small for collision checking'

		# no ignore collision objects
		self.collision_ignore_body_b_ids = []
		
		# ignore the agent's collision with these link ids of itself
		self.collision_ignore_link_a_ids = set(
			self.config.get('collision_ignore_link_a_ids', []))

		# task
		if self.config['task'] == 'navigate_goal_fixed':
			self.task = NavigateGoalFixedTask(self)
		else:
			self.task = None
			print("No such task defined")	


	def load_miscellaneous_variables(self):
		"""
		Load miscellaneous variables for book keeping
		"""
		self.current_step = 0
		self.collision_step = 0
		self.current_episode = 0
		self.collision_links = []

	def load(self):
		"""
		Load environment
		"""
		self.load_scene_robot()  # load robot and scene, use self load()
		self.load_task_setup()
		self.load_observation_space(self.task.task_obs_dim)
		self.load_action_space()
		self.load_miscellaneous_variables()	
		self.set_physics()

	def set_physics(self):
		# set floor friction coefficient
		self.set_floor_friction()

		print('--------------------------------')
		print('floor friction: %f'%(self.scene.get_floor_friction_coefficient()))
		print('--------------------------------')	

	# get collision links where bodyA=robot base link, bodyB=non-interactive objects
	def filter_collision_links(self):
		"""
		Filter out collisions that should be ignored

		:param collision_links: original collisions, a list of collisions
		:return: filtered collisions
		"""
		# only consider where bodyA=robot
		collision_links = list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0]))

		# 0-contactFlag, 1-bodyUniqueIdA, 2-bodyUniqueIdB, 3-linkIndexA, 4-linkIndexB
		filtered_collision_links = []

		for item in collision_links:
			# ignore collision where bodyA = ignored robot link (wheels)
			if item[3] in self.collision_ignore_link_a_ids:
				continue

			# ignore self collision where bodyA = not ignored robot link, bodyB = ignored robot link (wheels)
			#if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
			# ignore self collision where bodyA = not ignored robot link, bodyB = any robot link
			if item[2] == self.robots[0].robot_ids[0]:	
				continue

			# ignore collision between where bodyA = robot base, bodyB = interactive objects
			if item[2] in self.collision_ignore_body_b_ids:
				continue
			
			'''
			print('--------------------------------------------------------------')
			print('step: %d'%self.current_step)
			print('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(item[1], item[2], item[3], item[4]))
			'''
			# collision between where bodyA = robot base, bodyB = non interactive objects
			filtered_collision_links.append(item)

		return filtered_collision_links

	def run_simulation(self):
		"""
		Run simulation for one action timestep (same as one render timestep in Simulator class)

		:return: collision_links: collisions from last physics timestep
		"""
		self.simulator_step()
		

		return self.filter_collision_links()	

	# populate information into info
	def populate_info(self, info):
		"""
		Populate info dictionary with any useful information
		"""
		info['episode_length'] = self.current_step
		info['collision_step'] = self.collision_step # how many steps involve collisions
	
	def step(self, action):
		"""
		Apply robot's action.
		Returns the next state, reward, done and info,
		following OpenAI Gym's convention

		:param action: robot actions
		:return: state: next observation
		:return: reward: reward of this time step
		:return: done: whether the episode is terminated
		:return: info: info dictionary with any useful information
		"""
		self.current_step += 1

		if action is not None:
			self.robots[0].apply_action(action)

		# check collisions
		self.collision_links = self.run_simulation()
		self.collision_step += int(len(self.collision_links) > 0)
		
		state = self.get_state()
		info = {}

		reward, done, info, sub_reward = self.task.get_reward_termination(self, info)

		#print(sub_reward)

		# step task related variables
		self.task.step(self)

		self.populate_info(info)

		if done and self.automatic_reset:
			#info['last_observation'] = state  # useless in iGibson
			state = self.reset()

		return state, reward, done, info


	def reset_variables(self):
		"""
		Reset bookkeeping variables for the next new episode
		"""
		
		'''
		if self.collision_step > 0:
			print("total steps: %d"%(self.current_step))
			print("collision steps: %d"%(self.collision_step))
			print("-------------------------------------------------")
		'''
		self.current_episode += 1
		self.current_step = 0
		self.collision_step = 0
		self.collision_links = []


	def reset(self):
		"""
		Reset episode
		"""
		# move robot away from the scene
		self.robots[0].set_position([100.0, 100.0, 100.0])
		# no need to reset scene
		# reset agent and rewards
		self.task.reset_agent(self)
		self.simulator.sync()
		state = self.get_state()
		# reset other variables
		self.reset_variables()

		return state



if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'--config',
		'-c',
		help='which config file to use [default: use yaml files in examples/configs]', default='turtlebot_navigate.yaml')
	parser.add_argument('--mode',
						'-m',
						choices=['headless', 'gui', 'iggui'],
						default='headless',
						help='which mode for simulation (default: headless)')
	args = parser.parse_args()



	env = NavigateEnv(config_file=os.path.join(config_path, args.config),
					 mode=args.mode)

	
	step_time_list = []
	for episode in range(10):
		print("***********************************")
		print('Episode: {}'.format(episode))
		start = time.time()
		env.reset()
		for _ in range(200):  # 10 seconds
			action = env.action_space.sample()
			state, reward, done, info = env.step(action)
			#env.task.get_obj_goal_pos()
			#pos_distances, rot_distances = env.task.goal_distance()
			#print(pos_distances)
			#print(rot_distances)
			#print(env.observation_space)
			#print(env.state_space)
			#print(info)
			#print(state.shape)
			#print(state)
			#print('-----------------------------')
			#print(env.collision_step)
			#print('-------------------------------')
			#print('reward', reward)
			#print(state['task_obs'].shape)
			if done:
				break
	
		print('Episode finished after {} timesteps, took {} seconds.'.format(
			env.current_step, time.time() - start))
			
	env.close()
