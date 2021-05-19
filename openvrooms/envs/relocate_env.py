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
from openvrooms.robots.fetch_robot import Fetch

from gibson2.robots.husky_robot import Husky
from gibson2.robots.ant_robot import Ant
from gibson2.robots.humanoid_robot import Humanoid
from gibson2.robots.jr2_robot import JR2
from gibson2.robots.jr2_kinova_robot import JR2_Kinova
from gibson2.robots.freight_robot import Freight
#from gibson2.robots.fetch_robot import Fetch
from gibson2.robots.locobot_robot import Locobot


from openvrooms.tasks.relocate_goal_fixed_task import RelocateGoalFixedTask
from openvrooms.tasks.relocate_circle_task import RelocateCircleTask
from openvrooms.tasks.relocate_outside_circle_task import RelocateOutsideCircleTask
from openvrooms.scenes.relocate_scene import RelocateScene
from openvrooms.scenes.relocate_scene_different_objects import RelocateSceneDifferentObjects
from openvrooms.sensors.external_vision_sensor import ExternalVisionSensor
from openvrooms.config import *

from transforms3d.euler import euler2quat
from collections import OrderedDict
import argparse
import gym
import numpy as np
import pybullet as p
import time
import logging
import sys

#from gibson2.simulator import Simulator
from openvrooms.simulator.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

from openvrooms.utils.utils import l2_distance

from openvrooms.utils.utils import GRAVITY

class RelocateEnv(iGibsonEnv):
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
		'''
		super(RelocateEnv, self).__init__(config_file=config_file,
										 scene_id=scene_id,
										 mode = mode,
										 action_timestep=action_timestep,
										 physics_timestep=physics_timestep,
										 device_idx=device_idx,
										 render_to_tensor=render_to_tensor)
		'''
		
		self.config = parse_config(config_file)
		if scene_id is not None:
			self.config['scene_id'] = scene_id

		self.mode = mode

		self.action_timestep = self.config['action_timestep']
		#self.physics_timestep = self.config['physics_timestep']
		self.physics_timestep = 1 / 240.0

		# energy in reward function
		#self.energy_cost_scale = self.config.get('energy_cost_scale', 1.0)
		self.use_energy_cost = self.config.get('use_energy_cost')
		self.ratio_method = self.config.get('ratio_method')
		self.joint_level_energy = self.config.get('joint_level_energy')
		self.normalized_energy = self.config.get('normalized_energy')
		self.heuristic_succeed_episode_energy_min = float(self.config.get('heuristic_succeed_episode_energy_min', 0))
		self.heuristic_succeed_episode_energy_max = float(self.config.get('heuristic_succeed_episode_energy_max', 300))


		enable_shadow = self.config.get('enable_shadow', False)
		enable_pbr = self.config.get('enable_pbr', True)
		texture_scale = self.config.get('texture_scale', 1.0)

		self.image_shape = "HWC" #CHW, HWC

		settings = MeshRendererSettings(enable_shadow=enable_shadow,
										enable_pbr=enable_pbr,
										msaa=False,
										texture_scale=texture_scale)

		self.simulator = Simulator(mode=mode,
								   physics_timestep=self.physics_timestep,
								   render_timestep=self.action_timestep,
								   image_width=self.config.get(
									   'image_width', 128),
								   image_height=self.config.get(
									   'image_height', 128),
								   vertical_fov=self.config.get(
									   'vertical_fov', 90),
								   device_idx=device_idx,
								   render_to_tensor=render_to_tensor,
								   rendering_settings=settings,
								   external_camera_pos=self.config.get('external_camera_pos', [0, 0, 1.2]),
								   external_camera_view_direction=self.config.get('external_camera_view_direction', [1, 0, 0]),
								   normalized_energy=self.normalized_energy,
								   discrete_action_space=self.config.get('is_discrete', False), 
								   wheel_velocity=self.config.get('wheel_velocity', 1.0))

		self.load()								 

		self.automatic_reset = automatic_reset


		if self.use_energy_cost:
			print("Consider energy cost in success reward")
			if self.ratio_method == "paper":
				print("Use paper's method to compute ratio")
			elif self.ratio_method == "heuristic":
				print("Use heuristics to compute ratio")
			elif self.ratio_method == "history":
				print("Use running history to compute ratio")		
			else:
				print("Error: undefined ratio computing method")
		else:
			print("DO NOT consider energy cost in success reward")	
		print('--------------------------------')
		if self.config["use_goal_dist_reward"]:
			print("Use goal distance reward")
		else:
			print("DO NOT use goal distance reward")	
		print('--------------------------------')	

		
		
		if self.goal_conditioned:
			print("Goal conditioned")	
		else:
			print("Not goal conditioned")	
		print('--------------------------------')
	

		if self.config['scene'] == 'relocate_different_objects':
			if self.config['use_tier_reward']:
				print("Use tier reward")
			else:
				print("Not use tier reward")	
		print('--------------------------------')

		self.random_init_pose = self.config.get('random_init_pose', False)
		if self.random_init_pose:
			print("Randomize init pose")	
		else:
			print("Do NOT randomize init pose")	
		print('--------------------------------')

		self.swap = self.config.get('swap')
		if self.swap:
			print("Swap two box")
		else:
			print("Do NOT swap two box")

		print('--------------------------------')
		self.config_index = int(self.config.get('config_index', 0))
		print("Configuration index: %d"%(self.config_index))		

	def load_scene_robot(self):
		"""
		Import the scene and robot (but have not reset their poses)
		"""
		if self.config['scene'] == 'relocate':
			scene_id = self.config['scene_id']
			n_interactive_objects = self.config.get('obj_num', 1)
			if "multi_band" in scene_id:
				scene = RelocateScene(scene_id=scene_id, n_interactive_objects=n_interactive_objects, multi_band=True)
			else:	
				scene = RelocateScene(scene_id=scene_id, n_interactive_objects=n_interactive_objects)

			self.simulator.import_scene(scene, load_texture=self.config.get('load_texture', True))
			self.scene = scene
		elif self.config['scene'] == 'relocate_different_objects':
			scene_id = self.config['scene_id']
			n_interactive_objects = self.config.get('obj_num', 2)
			scene = RelocateSceneDifferentObjects(scene_id=scene_id, n_interactive_objects=n_interactive_objects, 
				material_names=self.config.get('obj_material', ['Material__wood_hemlock', 'Material__steel_oxydized_bright']))
			self.simulator.import_scene(scene, load_texture=self.config.get('load_texture', True))
			self.scene = scene
		else:
			raise Exception(
				'unknown scene type: {}'.format(self.config['scene']))

		self.load_robot()

	def set_interactive_obj_mass(self):
		obj_mass = np.array(self.config.get('obj_mass', [10]), dtype="float32")
		assert obj_mass.shape[0] == len(self.scene.interative_objects)

		for i, obj in enumerate(self.scene.interative_objects):
			obj.set_mass(obj_mass[i])

	def set_floor_friction(self):
		floor_friction = np.array(self.config.get('floor_friction', [0.5]), dtype="float32")
		self.scene.set_floor_friction_coefficient(mu=floor_friction)		

	def set_physics(self):
		# set interactive objects weights
		self.set_interactive_obj_mass()

		# set floor friction coefficient
		self.set_floor_friction()

		print('-------------- object mass ------------------')
		for obj in self.scene.interative_objects:
			print(obj.get_mass())
		print('-------------- object material ------------------')
		for obj in self.scene.interative_objects:
			print(obj.get_material())
		print('--------------------------------')
		if self.scene.multi_band:
			print('floor friction: %s'%(str(self.scene.get_floor_friction_coefficient())))
		else:
			print('floor friction: %f'%(self.scene.get_floor_friction_coefficient()))
		print('--------------------------------')


	
	def load_robot(self):
		# load robot
		if self.config['robot'] == 'Turtlebot':
			robot = Turtlebot(self.config)
		elif self.config['robot'] == 'Husky':
			robot = Husky(self.config)
		elif self.config['robot'] == 'Ant':
			robot = Ant(self.config)
		elif self.config['robot'] == 'Humanoid':
			robot = Humanoid(self.config)
		elif self.config['robot'] == 'JR2':
			robot = JR2(self.config)
		elif self.config['robot'] == 'JR2_Kinova':
			robot = JR2_Kinova(self.config)
		elif self.config['robot'] == 'Freight':
			robot = Freight(self.config)
		elif self.config['robot'] == 'Fetch':
			robot = Fetch(self.config)
		elif self.config['robot'] == 'Locobot':
			robot = Locobot(self.config)
		else:
			raise Exception(
				'unknown robot type: {}'.format(self.config['robot']))

		self.robots = [robot]
		for robot in self.robots:
			self.simulator.import_robot(robot)	

	def load_task_setup(self):
		"""
		Load task setup
		"""
		self.initial_pos_z_offset = self.config.get(
			'initial_pos_z_offset', 0.1)
		# s = 0.5 * G * (t ** 2)
		drop_distance = 0.5 * GRAVITY * (self.action_timestep ** 2)
		assert drop_distance < self.initial_pos_z_offset, \
			'initial_pos_z_offset is too small for collision checking'

		# ignore the agent's collision with these body ids
		self.collision_ignore_body_b_ids = set(
			self.config.get('collision_ignore_body_b_ids', []))
		
		# ignore the agent's collision with these link ids of itself
		self.collision_ignore_link_a_ids = set(
			self.config.get('collision_ignore_link_a_ids', []))

		# task
		if self.config['task'] == 'relocate_goal_fixed':
			self.task = RelocateGoalFixedTask(self)
		elif self.config['task'] == 'relocate_circle':
			self.task = RelocateCircleTask(self)
		elif self.config['task'] == 'relocate_outside_circle':
			self.task = RelocateOutsideCircleTask(self)		
		else:
			self.task = None
			print("No such task defined")	


	

	def load_action_space(self):
		"""
		Load action space
		"""
		agent = self.robots[0]
		self.action_space = agent.action_space
		print("-----------------------------------")
		print("Action space: ")
		if agent.is_discrete:
			print("Discrete: dim = %d"%(agent.action_dim))
			print("%d actions"%(agent.action_space.n))
			for action in agent.action_list:
				print(str(action))
		else:
			print("Continuous: dim = %d"%(agent.action_dim))
			print("Action low: %s"%(agent.action_low))
			print("Action high: %s"%(agent.action_high))
		print("-----------------------------------")

	def load_miscellaneous_variables(self):
		"""
		Load miscellaneous variables for book keeping
		"""
		self.current_step = 0 # per episode
		self.non_interactive_collision_step = 0 # per episode
		self.interactive_collision_step = 0 # per episode
		self.current_episode = 0

		self.current_episode_robot_energy_cost = 0.0 # per episode
		self.current_episode_pushing_energy_translation = 0.0 # per episode
		self.current_episode_pushing_energy_rotation = 0.0 # per episode

		#self.current_succeed_episode_robot_energy_cost = 0.0 # per episode
		#self.current_succeed_episode_pushing_energy_translation = 0.0 # per episode
		#self.current_succeed_episode_pushing_energy_rotation = 0.0 # per episode

		self.non_interactive_collision_links = [] # per step
		self.interactive_collision_links = [] # per step

		## record min and max energy among all successful episodes
		self.max_succeed_episode_robot_energy_cost = 0.
		self.max_succeed_episode_pushing_energy_cost = 0.
		self.min_succeed_episode_robot_energy_cost = np.inf
		self.min_succeed_episode_pushing_energy_cost = np.inf

	def load(self):
		"""
		Load environment
		"""
		self.load_scene_robot()  # load robot and scene, use self load()
		self.load_task_setup()
		
		self.goal_conditioned = self.config.get('goal_conditioned', False)
		if self.goal_conditioned:
			task_obs_dim = self.task.task_obs_dim+self.task.obj_num*12
		else:
			task_obs_dim = self.task.task_obs_dim+self.task.obj_num*6

		self.load_observation_space(task_obs_dim)
		self.load_action_space()
		self.load_miscellaneous_variables()
		self.set_physics()

	def load_observation_space(self, task_obs_dim):
		"""
		Load observation space
		"""
		# Three modes: task_obs, vision, scan, only one of them can exist
		# Sensors can be vision or scan 
		# output can have task_objs or vision modalities or scan modalities
		# vision and scan can have multiple modalities
		self.output = self.config['output']
		self.image_width = self.config.get('image_width', 128)
		self.image_height = self.config.get('image_height', 128)
		observation_space = OrderedDict()
		sensors = OrderedDict()
		vision_modalities = []
		scan_modalities = []


		# task obs
		if 'task_obs' in self.output:
			observation_space['task_obs'] = self.build_obs_space(
				shape=(task_obs_dim,), low=-np.inf, high=np.inf)
		# vision modalities	
		if 'rgb' in self.output:
			if self.image_shape == "CHW":
				observation_space['rgb'] = self.build_obs_space(
					shape=(3, self.image_height, self.image_width),
					low=0.0, high=1.0)
			else:
				observation_space['rgb'] = self.build_obs_space(
					shape=(self.image_height, self.image_width, 3),
					low=0.0, high=1.0)

			vision_modalities.append('rgb')
		if 'depth' in self.output:
			if self.image_shape == "CHW":
				observation_space['depth'] = self.build_obs_space(
				shape=(1, self.image_height, self.image_width),
				low=0.0, high=1.0)
			else:
				observation_space['depth'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 1),
				low=0.0, high=1.0)	

			vision_modalities.append('depth')
		if 'pc' in self.output:
			if self.image_shape == "CHW":
				observation_space['pc'] = self.build_obs_space(
				shape=(3, self.image_height, self.image_width),
				low=-np.inf, high=np.inf)
			else:
				observation_space['pc'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 3),
				low=-np.inf, high=np.inf)	

			vision_modalities.append('pc')
		if 'optical_flow' in self.output:
			if self.image_shape == "CHW":
				observation_space['optical_flow'] = self.build_obs_space(
				shape=(2, self.image_height, self.image_width),
				low=-np.inf, high=np.inf)
			else:
				observation_space['optical_flow'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 2),
				low=-np.inf, high=np.inf)	

			vision_modalities.append('optical_flow')
		if 'scene_flow' in self.output:
			if self.image_shape == "CHW":
				observation_space['scene_flow'] = self.build_obs_space(
				shape=(3, self.image_height, self.image_width),
				low=-np.inf, high=np.inf)
			else:
				observation_space['scene_flow'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 3),
				low=-np.inf, high=np.inf)	

			vision_modalities.append('scene_flow')
		if 'normal' in self.output:
			if self.image_shape == "CHW":
				observation_space['normal'] = self.build_obs_space(
				shape=(3, self.image_height, self.image_width),
				low=-np.inf, high=np.inf)
			else:
				observation_space['normal'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 3),
				low=-np.inf, high=np.inf)

			vision_modalities.append('normal')
		if 'seg' in self.output:
			if self.image_shape == "CHW":
				observation_space['seg'] = self.build_obs_space(
				shape=(1, self.image_height, self.image_width),
				low=0.0, high=1.0)
			else:
				observation_space['seg'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 1),
				low=0.0, high=1.0)	

			vision_modalities.append('seg')
		if 'rgb_filled' in self.output:  # use filler
			if self.image_shape == "CHW":
				observation_space['rgb_filled'] = self.build_obs_space(
				shape=(3, self.image_height, self.image_width),
				low=0.0, high=1.0)
			else:
				observation_space['rgb_filled'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 3),
				low=0.0, high=1.0)	

			vision_modalities.append('rgb_filled')
		
		# scan modalities		
		if 'scan' in self.output:
			self.n_horizontal_rays = self.config.get('n_horizontal_rays', 128)
			self.n_vertical_beams = self.config.get('n_vertical_beams', 1)
			assert self.n_vertical_beams == 1, 'scan can only handle one vertical beam for now'
			observation_space['scan'] = self.build_obs_space(
				shape=(self.n_horizontal_rays * self.n_vertical_beams, 1),
				low=0.0, high=1.0)
			scan_modalities.append('scan')
		if 'occupancy_grid' in self.output:
			self.grid_resolution = self.config.get('grid_resolution', 128)

			if self.image_shape == "CHW":
				self.occupancy_grid_space = gym.spaces.Box(low=0.0,
													   high=1.0,
													   shape=(1, self.grid_resolution,
															  self.grid_resolution))
			else:
				self.occupancy_grid_space = gym.spaces.Box(low=0.0,
													   high=1.0,
													   shape=(self.grid_resolution,
															  self.grid_resolution, 1))

			observation_space['occupancy_grid'] = self.occupancy_grid_space
			scan_modalities.append('occupancy_grid')

		# create sensors
		if len(vision_modalities) > 0:
			third_person_view = self.config.get("third_person_view", True)
			# third person view
			if third_person_view:
				sensors['vision'] = ExternalVisionSensor(self, vision_modalities, camera_pos=self.config.get('external_camera_pos', [0, 0, 1.2]),
								   camera_view_direction=self.config.get('external_camera_view_direction', [1, 0, 0]))
			# first person view
			else:
				sensors['vision'] = VisionSensor(self, vision_modalities)

		if len(scan_modalities) > 0:
			sensors['scan_occ'] = ScanSensor(self, scan_modalities)

		self.sensors = sensors
		self.vision_modalities = vision_modalities
		self.scan_modalities = scan_modalities
		
		# create observation space
		#self.observation_space = gym.spaces.Dict(observation_space)
		#self.observation_space = self.combine_vision_observation_space(vision_modalities, observation_space)
		
		if 'task_obs' in self.output:
			self.observation_space = observation_space['task_obs']
		elif 'rgb' in self.output:
			self.observation_space = observation_space['rgb']

		
		
	
	# can only combine when the each modality has the same image size
	def combine_vision_observation_space(self, vision_modalities, observation_space):
		channel_num = 0
		low = np.inf
		high = -np.inf
		for modal in vision_modalities:
			channel_num += observation_space[modal].shape[0]
			cur_low = np.amin(observation_space[modal].low)
			cur_high = np.amax(observation_space[modal].high)

			if cur_low < low:
				low = cur_low

			if cur_high > high:
				high = cur_high	

		# [H,W,C]
		if self.image_shape == "HWC":
			return self.build_obs_space(
				shape=(self.image_height, self.image_width, channel_num),
				low=low, high=high)
		# [C,H,W]	
		else:	
			return self.build_obs_space(
				shape=(channel_num, self.image_height, self.image_width),
				low=low, high=high)
	# to use all and make it gym compatible, ensure that the output is np.array
	# right now, learning can only handle single modal
	def get_state(self):
		"""
		Get the current observation

		:param collision_links: collisions from last physics timestep
		:return: observation as a dictionary
		"""
		state = OrderedDict()
		
		# state 
		if 'task_obs' in self.output:
			state['task_obs'] = self.task.get_task_obs(self)

		# observation
		if 'vision' in self.sensors:
			vision_obs = self.sensors['vision'].get_obs(self)
			for modality in vision_obs:
				if self.image_shape == "CHW":
					state[modality] = np.transpose(vision_obs[modality], (2,0,1))
				else:
					state[modality] = vision_obs[modality]	

		if 'scan_occ' in self.sensors:
			scan_obs = self.sensors['scan_occ'].get_obs(self)

			for modality in scan_obs:
				if modality == 'occupancy_grid':
					if self.image_shape == "CHW":
						state[modality] = np.transpose(scan_obs[modality], (2,0,1))
					else:
						state[modality]	= scan_obs[modality]
				else:	
					state[modality] = scan_obs[modality]
		
		#return self.combine_vision_observation(self.vision_modalities, state)

		#return state
		# single state modal as np.array
		if 'task_obs' in self.output:
			return state['task_obs']
		elif 'rgb' in self.output:
			return state['rgb']
		
	# can only combine when the each modality has the same image size
	def combine_vision_observation(self, vision_modalities, state):
		combined_state = state[vision_modalities[0]]
		
		
		for modal in vision_modalities[1:]:
			# [H,W,C]
			if self.image_shape == "HWC":
				combined_state = np.concatenate((combined_state, state[modal]), axis=2)
			# [C,H,W]	
			else:	
				combined_state = np.concatenate((combined_state, state[modal]), axis=0)

		#print(combined_state.shape)
		return combined_state	
	

	# run physics engine simulator for n steps (run one action step)
	def run_simulation(self):
		"""
		Run simulation for one action timestep (same as one render timestep in Simulator class)

		:return: collision_links: collisions from last physics timestep
		"""
		
		# call simulator.step()
		# run one action step
		self.simulator_step()

		# check collision at the end of n physics simulator steps

		# only consider where bodyA=robot
		non_interactive_collision_links, interactive_collision_links = self.filter_collision_links()

		'''
		bodyB_robot = list(p.getContactPoints(bodyB=self.robots[0].robot_ids[0]))
		if len(bodyB_robot) > 0:
			print("step %d: Robot is bodyB"%(self.current_step))
			for item in bodyB_robot:
				print('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(item[1], item[2], item[3], item[4]))
			print("----------------------------------")

		if len(collision_links) > 0:
			print("step %d: Robot is bodyA"%(self.current_step))
			for item in collision_links:
				print('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(item[1], item[2], item[3], item[4]))
			print("----------------------------------")		
		'''	

		# get collision where bodyA=interactive_object, bodyB=non_interactive_object or interactive_object
		negative_collisions = self.filter_interactive_collision_links()
		non_interactive_collision_links.extend(negative_collisions)

		'''
		if len(non_interactive_collision_links) > 0:
			for item in non_interactive_collision_links:
				print('--------------------------------------------------------------')
				print('step: %d'%self.current_step)
				print('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(item[1], item[2], item[3], item[4]))
		'''
		'''
		if len(interactive_collision_links) > 0:
			for item in interactive_collision_links:
				print('--------------------------------------------------------------')
				print('step: %d'%self.current_step)
				print('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(item[1], item[2], item[3], item[4]))		
		'''

		# get robot energy of current action step after stepping physics simulator
		current_step_robot_energy_cost = self.simulator.robot_energy_cost

		
		return non_interactive_collision_links, interactive_collision_links, current_step_robot_energy_cost

	# get collision where bodyA=interactive_object, bodyB=non_interactive_object or interactive_object
	def filter_interactive_collision_links(self):
		negative_collisions = []
		collision_ignore_body_b_ids_list = list(self.collision_ignore_body_b_ids)

		for i, object_id in enumerate(collision_ignore_body_b_ids_list):
			collision_links = list(p.getContactPoints(bodyA=object_id))
			for item in collision_links:
				# ignore collision between interactive objects and robot
				if item[2] == self.robots[0].robot_ids[0]:
					continue

				# ignore collisions between interactive objects and floor
				if self.scene.multi_band:
					if item[2] in self.scene.floor_id:
						#print("collsion: box and floor: %d"%(item[2]))
						continue
				else:
					if item[2] == self.scene.floor_id:
						continue

				# collide with another interactive objects
				if item[2] in collision_ignore_body_b_ids_list[i+1:]:
					negative_collisions.append(item)

				# collide with static objects except floor
				negative_collisions.append(item)

				'''
				print('--------------------------------------------------------------')
				print('step: %d'%self.current_step)
				print('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(item[1], item[2], item[3], item[4]))
				'''
				
		return negative_collisions

	def get_interactive_obj_physics(self, obj):
		object_urdf_id = obj.body_id
		collisions = list(p.getContactPoints(bodyA=object_urdf_id))
		# 0-contactFlag, 1-bodyUniqueIdA, 2-bodyUniqueIdB, 3-linkIndexA, 4-linkIndexB
		floor_collisions = []
		for item in collisions:
			if self.scene.multi_band:
				# collision where bodyA = wheels, bodyB = floor
				if item[2] in self.scene.floor_id:
					#print("collsion: box and floor: %d"%(item[2]))
					floor_collisions.append(item)
			# single floor		
			else:
				# collision where bodyA = wheels, bodyB = floor
				if item[2] == self.scene.floor_id:
					floor_collisions.append(item)

		# average floor coefficients if there are more than one floors
		if len(floor_collisions) > 0:
			floor_friction_coefficient = 0
			for floor_collision in floor_collisions:
				floor_urdf_id = floor_collision[2]	
				floor_friction_coefficient += p.getDynamicsInfo(floor_urdf_id, -1)[1]

			floor_friction_coefficient /= float(len(floor_collisions))	
		else:
			print("Error: no floor is contacting with object %d, use default floor"%(object_urdf_id))	
			if self.scene.multi_band:
				floor_urdf_id = self.scene.floor_id[0]
			else:
				floor_urdf_id = self.scene.floor_id	

			floor_friction_coefficient = p.getDynamicsInfo(floor_urdf_id, -1)[1]	

		object_mass = obj.get_mass()

		current_orn_z = obj.get_orientation_z()

		current_pos_xy = obj.get_xy_position()

		obj_x_width, obj_y_width = obj.get_xy_dimension()

		return floor_friction_coefficient, object_mass, current_pos_xy, current_orn_z, obj_x_width, obj_y_width


	# get collision links with robot base link
	# get collision links where bodyA=robot base link, bodyB=non-interactive objects and interactive objects respectively
	def filter_collision_links(self):
		"""
		Filter out collisions that should be ignored

		:param collision_links: original collisions, a list of collisions
		:return: filtered collisions
		"""

		collision_links = list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0]))
		# 0-contactFlag, 1-bodyUniqueIdA, 2-bodyUniqueIdB, 3-linkIndexA, 4-linkIndexB
		non_interactive_collision_links = []
		interactive_collision_links = []
		

		for item in collision_links:
			# ignore collision where bodyA = ignored robot link (wheels)
			if item[3] in self.collision_ignore_link_a_ids:
				continue
				
			# keep ignore wheel collisions: ignore self collision where bodyA = not ignored robot link, bodyB = ignored robot link (wheels)
			#if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
			# ignore self collision where bodyA = not ignored robot link, bodyB = any robot link
			if item[2] == self.robots[0].robot_ids[0]:
				continue

			# collision between where bodyA = robot base, bodyB = interactive objects
			if item[2] in self.collision_ignore_body_b_ids:
				interactive_collision_links.append(item)
			# collision between where bodyA = robot base, bodyB = non interactive objects	
			else:
				#print("***********************************")
				#print("non-interactive collision: %d"%(item[2]))
				'''
				print('--------------------------------------------------------------')
				print('step: %d'%self.current_step)
				print('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(item[1], item[2], item[3], item[4]))
				'''
				non_interactive_collision_links.append(item)

		return non_interactive_collision_links, interactive_collision_links

	def calc_pushing_energy_trans(self, pos1, pos2, mass, coeff):
		pos1 = np.array(pos1).flatten()
		pos2 = np.array(pos2).flatten()

		# note that usually pos1 != pos2 even if object did not move
		#print(np.array_equal(pos1, pos2))
		#print(pos1)
		#print(pos2)
		#print('---------------------------------------')

		assert len(pos1) == len(pos2) == 2, f"[calc_pushing_energy_trans]Error: invalid position vector size: pos1={len(pos1)} and pos2={len(pos2)}!"
	
		# igibson set gravity as 9.8
		return abs(coeff * mass) * GRAVITY * np.linalg.norm(pos1 - pos2)

	def calc_pushing_energy_rot(self, orn1: float, orn2: float, mass: float, coeff: float, len_x: float, len_y: float)-> float:
		# calc angle rotated
		theta = abs(orn2 - orn1)

		# calc gravity per unit area
		rho = abs(mass * GRAVITY) / abs(len_x * len_y)

		return coeff * rho * theta * self.calc_integral(len_x, len_y)

	def calc_integral(self, X: float, Y: float) -> float:
		X = abs(X)
		Y = abs(Y)

		# calc trigonometric functions
		tan = Y / X
		cos = X / np.sqrt(X**2 + Y**2)
		sin = Y / np.sqrt(X**2 + Y**2)

		# calc \integral 1/cos(\theta)^3 d\theta, 0 to \phi
		A = 1/2 * (tan/cos + np.log(tan + 1/cos))

		# calc \integral 1/sin(\theta)^3 d\theta, \phi to \pi/2
		B = 1/2 * ( 1/tan * 1/sin - np.log( np.sqrt((1 - cos)/(1 + cos)) ) )

		return (X**3)/6 * A + (Y**3)/6 * B	

	# populate information into info
	def populate_info(self, info):
		"""
		Populate info dictionary with any useful information
		"""
		info['episode_length'] = self.current_step
		info['non_interactive_collision_step'] = self.non_interactive_collision_step # how many steps involve collision with non-interactive objects
		info['interactive_collision_step'] = self.interactive_collision_step # how many steps involve collision with interactive objects

	def compute_pushing_energy_per_action_step(self, prev_obj_pos_xy, prev_obj_orn_z):
		current_step_pushing_energy_translation = 0
		current_step_pushing_energy_rotation = 0

		for i, obj in enumerate(self.scene.interative_objects):
			floor_friction_coefficient, object_mass, current_pos_xy, current_orn_z, obj_x_width, obj_y_width = self.get_interactive_obj_physics(obj)
			obj_pushing_energy_translation = self.calc_pushing_energy_trans(pos1=prev_obj_pos_xy[i], pos2=current_pos_xy, mass=object_mass, coeff=floor_friction_coefficient)
			#print(obj_pushing_energy_translation)
			obj_pushing_energy_rotation = self.calc_pushing_energy_rot(orn1=prev_obj_orn_z[i], orn2=current_orn_z, mass=object_mass, coeff=floor_friction_coefficient, len_x=obj_x_width, len_y=obj_y_width)

			current_step_pushing_energy_translation += obj_pushing_energy_translation
			current_step_pushing_energy_rotation += obj_pushing_energy_rotation

		return  current_step_pushing_energy_translation, current_step_pushing_energy_rotation

	# only get called when the episode is success
	# lower energy --> lower ratio
	# ratio: [0,1]
	# three methods
	def compute_energy_ratio(self):
		# running history
		if self.ratio_method == "history":
			if self.joint_level_energy: # normalized or not
				if self.max_succeed_episode_robot_energy_cost == self.min_succeed_episode_robot_energy_cost:
					ratio = 0
				else:	
					ratio = (self.current_episode_robot_energy_cost - self.min_succeed_episode_robot_energy_cost) / float(self.max_succeed_episode_robot_energy_cost - self.min_succeed_episode_robot_energy_cost)
			else:
				if self.max_succeed_episode_pushing_energy_cost == self.min_succeed_episode_pushing_energy_cost:
					ratio = 0
				else:	
					current_episode_pushing_energy_cost = self.current_episode_pushing_energy_translation + self.current_episode_pushing_energy_rotation
					ratio = (current_episode_pushing_energy_cost - self.min_succeed_episode_pushing_energy_cost) / float(self.max_succeed_episode_pushing_energy_cost - self.min_succeed_episode_pushing_energy_cost)
		# heuristics 
		elif self.ratio_method == "heuristic":
			if self.joint_level_energy: # normalized or not
				ratio = (self.current_episode_robot_energy_cost - self.heuristic_succeed_episode_energy_min) / float(self.heuristic_succeed_episode_energy_max - self.heuristic_succeed_episode_energy_min)
			else:
				current_episode_pushing_energy_cost = self.current_episode_pushing_energy_translation + self.current_episode_pushing_energy_rotation
				ratio = (current_episode_pushing_energy_cost - self.heuristic_succeed_episode_energy_min) / float(self.heuristic_succeed_episode_energy_max - self.heuristic_succeed_episode_energy_min)	
		# paper's method		
		elif self.ratio_method == "paper":
			assert self.joint_level_energy == True, "[relocate_env] Energy ratio computed by paper's method is only supported by joint level energy cost!"
			assert self.normalized_energy == True, "[relocate_env] Energy ratio computed by paper's method is only supported when joint level energy is normalized!"
			
			physics_simulation_steps = int(self.config.get('max_step')) * int(self.action_timestep / self.physics_timestep)
			ratio = self.current_episode_robot_energy_cost / float(physics_simulation_steps)	
		else:
			print("Error: undefined ratio computing method")
			return 0

		return ratio	

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

		
		# before step 
		prev_obj_pos_xy = list()
		prev_obj_orn_z = list()
		for obj in self.scene.interative_objects:
			prev_obj_pos_xy.append(obj.get_xy_position())
			prev_obj_orn_z.append(obj.get_orientation_z())
			

		# step simulator, check collisions, compute current_step_robot_energy_cost
		non_interactive_collision_links, interactive_collision_links, current_step_robot_energy_cost = self.run_simulation()

		# after step
		# used by reward_termination collision
		self.non_interactive_collision_links = non_interactive_collision_links
		self.interactive_collision_links = interactive_collision_links

		self.non_interactive_collision_step += int(len(non_interactive_collision_links) > 0)
		self.interactive_collision_step += int(len(interactive_collision_links) > 0)


		# accumulate robot energy cost at this step
		self.current_episode_robot_energy_cost += current_step_robot_energy_cost
		
		#print('Energy cost: %f'%(self.robot_energy_cost_cur_step * self.energy_cost_scale))
		#print('Action: %s'%(action))
		#if len(interactive_collision_links) > 0:
		#	print('Push')
		#print('--------------------------')

		# accumulate pushing energy cost at this step
		current_step_pushing_energy_translation, current_step_pushing_energy_rotation = self.compute_pushing_energy_per_action_step(prev_obj_pos_xy, prev_obj_orn_z)
		self.current_episode_pushing_energy_translation +=  current_step_pushing_energy_translation
		self.current_episode_pushing_energy_rotation += current_step_pushing_energy_rotation

		state = self.get_state()
		info = {}

		if self.config['scene'] == 'relocate_different_objects':
			if self.config['use_tier_reward'] == False:
				reward, done, info, sub_reward = self.task.get_reward_termination(self, info)
			else:	
				reward, done, info, sub_reward = self.task.get_reward_termination_different_objects(self, info)
		else:
			reward, done, info, sub_reward = self.task.get_reward_termination(self, info)		

		# if succeed, update min and max energy among all successful episodes
		if info['success']:
			# compute current episode pushing energy
			current_episode_pushing_energy_cost = self.current_episode_pushing_energy_translation + self.current_episode_pushing_energy_rotation
			
			self.max_succeed_episode_robot_energy_cost = max(self.max_succeed_episode_robot_energy_cost, self.current_episode_robot_energy_cost)
			self.max_succeed_episode_pushing_energy_cost = max(self.max_succeed_episode_pushing_energy_cost, current_episode_pushing_energy_cost)
			self.min_succeed_episode_robot_energy_cost = min(self.min_succeed_episode_robot_energy_cost, self.current_episode_robot_energy_cost)
			self.min_succeed_episode_pushing_energy_cost = min(self.min_succeed_episode_pushing_energy_cost, current_episode_pushing_energy_cost)

			#self.current_succeed_episode_robot_energy_cost = self.current_episode_robot_energy_cost
			#self.current_succeed_episode_pushing_energy_translation = self.current_episode_pushing_energy_translation
			#self.current_succeed_episode_pushing_energy_rotation = self.current_episode_pushing_energy_rotation

		# consider energy cost in reward function when succeed
		# make sure that current_episode_energy, max_succeed and min_succeed are updated before ratio
		if info['success'] and self.use_energy_cost:
			ratio = self.compute_energy_ratio()
			reward = reward * (1 - ratio)
		
		#print(sub_reward)

		# step task related variables
		self.task.step(self)

		self.populate_info(info)

		if done and self.automatic_reset:
			#info['last_observation'] = state  # useless in iGibson
			state = self.reset()

		return state, reward, done, info

	# C,H,W
	# for gym compatibility
	def render(self, mode='rgb'):
		if 'vision' in self.sensors:
			vision_obs = self.sensors['vision'].get_obs(self)
			return vision_obs[mode]
		else:
			raise Exception('Missing vision sensor')		
	# return contact points with body_id
	def check_collision(self, body_id):
		"""
		Check with the given body_id has any collision after one simulator step

		:param body_id: pybullet body id
		:return: whether the given body_id has no collision
		"""
		self.simulator_step()
		collisions = list(p.getContactPoints(bodyA=body_id))

		if logging.root.level <= logging.DEBUG:  # Only going into this if it is for logging --> efficiency
			for item in collisions:
				logging.debug('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(
					item[1], item[2], item[3], item[4]))
		
		

		return len(collisions) == 0

	def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
		"""
		Reset position and orientation for the robot or the object

		:param obj: an instance of robot or object
		:param pos: position
		:param orn: orientation (euler angles: rotatation around x,y,z axis)
		:param offset: z offset
		"""
		if orn is None:
			orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

		if offset is None:
			offset = self.initial_pos_z_offset

		is_robot = isinstance(obj, BaseRobot)
		body_id = obj.robot_ids[0] if is_robot else obj.body_id
		# first set the correct orientation
		# param orn: quaternion in xyzw
		# quatToXYZW: convert quaternion from arbitrary sequence to XYZW (pybullet convention)
		obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), 'wxyz'))
		# compute stable z based on this orientation
		stable_z = stable_z_on_aabb(body_id, [pos, pos])
		# change the z-value of position with stable_z + additional offset
		# in case the surface is not perfect smooth (has bumps)
		obj.set_position([pos[0], pos[1], stable_z + offset])

	# Reset position and orientation for the robot or the object without z offset
	# orn: orientation (euler angles: rotatation around x,y,z axis)
	def set_pos_orn(self, obj, pos, orn=None): 	
		if orn is None:
			orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

		is_robot = isinstance(obj, BaseRobot)
		body_id = obj.robot_ids[0] if is_robot else obj.body_id

		obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), 'wxyz'))	


	def test_valid_position(self, obj, pos, orn=None):
		"""
		Test if the robot or the object can be placed with no collision

		:param obj: an instance of robot or object
		:param pos: position
		:param orn: orientation
		:return: validity
		"""
		is_robot = isinstance(obj, BaseRobot)

		self.set_pos_orn_with_z_offset(obj, pos, orn)
		#self.set_pos_orn(obj, pos, orn)

		if is_robot:
			obj.robot_specific_reset()
			obj.keep_still()

		body_id = obj.robot_ids[0] if is_robot else obj.body_id
		has_collision = self.check_collision(body_id)
		return has_collision

	# land object or robot with an initial height above the floor
	def land(self, obj, pos, orn):
		"""
		Land the robot or the object onto the floor, given a valid position and orientation

		:param obj: an instance of robot or object
		:param pos: position
		:param orn: orientation
		"""
		is_robot = isinstance(obj, BaseRobot)

		#self.set_pos_orn(obj, pos, orn)
		self.set_pos_orn_with_z_offset(obj, pos, orn)

		if is_robot:
			obj.robot_specific_reset()
			obj.keep_still()

		body_id = obj.robot_ids[0] if is_robot else obj.body_id

		
		land_success = False
		# land for maximum 1 second, should fall down ~5 meters
		max_simulator_step = int(1.0 / self.action_timestep)
		for _ in range(max_simulator_step):
			self.simulator_step()
			if len(p.getContactPoints(bodyA=body_id)) > 0:
				land_success = True
				break

		if not land_success:
			print("WARNING: Failed to land")
		
		if is_robot:
			obj.robot_specific_reset()
			obj.keep_still()

		# keep still for some timesteps	
		warm_up_step = 50
		for _ in range(warm_up_step):
			self.simulator_step()


	def reset_variables(self):
		"""
		Reset bookkeeping variables for the next new episode
		"""
		
		#if self.interactive_collision_step > 0:
		#	print("total steps: %d"%(self.current_step))
		#	print("non interactive collision steps: %d"%(self.non_interactive_collision_step))
		#	print("interactive collision steps: %d"%(self.interactive_collision_step))
		
		#print('------------------------------------')
		#print(self.non_interactive_collision_step)

		'''
		if self.non_interactive_collision_step > 0:
			print("total steps: %d"%(self.current_step))
			print("non interactive collision steps: %d"%(self.non_interactive_collision_step))
			print("-------------------------------------------------")
		'''
		self.current_episode += 1
		self.current_step = 0 # per episode
		self.non_interactive_collision_step = 0 # per episode
		self.interactive_collision_step = 0 # per episode

		self.current_episode_robot_energy_cost = 0.0 # per episode
		self.current_episode_pushing_energy_translation = 0.0 # per episode
		self.current_episode_pushing_energy_rotation = 0.0 # per episode

		#self.current_succeed_episode_robot_energy_cost = 0.0 # per episode
		#self.current_succeed_episode_pushing_energy_translation = 0.0 # per episode
		#self.current_succeed_episode_pushing_energy_rotation = 0.0 # per episode

		self.non_interactive_collision_links = [] # per step
		self.interactive_collision_links = [] # per step


	def reset(self):
		"""
		Reset episode
		"""
		# move robot away from the scene
		self.robots[0].set_position([100.0, 100.0, 100.0])

		if self.random_init_pose == False:
			# reset scene
			self.task.reset_scene(self)
			# reset agent and rewards
			self.task.reset_agent(self)
		# randomize agent and box pose	
		else:
			self.task.reset_scene_agent_random(self)

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
		help='which config file to use [default: use yaml files in examples/configs]', default='fetch_relocate_short.yaml')
	parser.add_argument('--mode',
						'-m',
						choices=['headless', 'gui', 'iggui'],
						default='headless',
						help='which mode for simulation (default: headless)')
	args = parser.parse_args()


	#sys.stdout = open('/home/meng/ray_results/output.txt', 'w')

	env = RelocateEnv(config_file=os.path.join(config_path, args.config),
					 mode=args.mode)

	
	step_time_list = []
	for episode in range(40):
		print("***********************************")
		print('Episode: {}'.format(episode))
		start = time.time()
		env.reset()
		for _ in range(100):  # 10 seconds
			#action = env.action_space.sample()
			action = 3
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
			print('-----------------------------')
			print('reward', reward)
			print('-------------------------------')
			#print(state['task_obs'].shape)
			if done:
				break
			#print('...')
		print('Episode robot output energy: %f'%(env.current_episode_robot_energy_cost))
		print('Episode pushing energy (translation): %f'%(env.current_episode_pushing_energy_translation))
		print('Episode pushing energy (rotation): %f'%(env.current_episode_pushing_energy_rotation))	
		#print('Episode energy cost (normalized): %f'%(env.current_episode_robot_energy_cost/float(400.0)))
		#print('Episode energy cost: %f'%(env.current_episode_robot_energy_cost/float(env.current_step)))
		print('Episode finished after {} timesteps, took {} seconds.'.format(
			env.current_step, time.time() - start))
	
	env.close()

	#sys.stdout.close()
	
	