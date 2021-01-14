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


from openvrooms.tasks.relocate_point_goal_fixed_task import RelocatePointGoalFixedTask
from openvrooms.scenes.relocate_scene import RelocateScene
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

#from gibson2.simulator import Simulator
from openvrooms.simulator.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings

class RelocateEnv(iGibsonEnv):
	"""
	iGibson Environment (OpenAI Gym interface)
	"""

	def __init__(
		self,
		config_file,
		scene_id=None,
		mode='headless',
		action_timestep=1 / 10.0,
		physics_timestep=1 / 240.0,
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
		self.action_timestep = action_timestep
		self.physics_timestep = physics_timestep
		self.texture_randomization_freq = self.config.get(
			'texture_randomization_freq', None)
		self.object_randomization_freq = self.config.get(
			'object_randomization_freq', None)
		self.object_randomization_idx = 0
		self.num_object_randomization_idx = 10

		enable_shadow = self.config.get('enable_shadow', False)
		enable_pbr = self.config.get('enable_pbr', True)
		texture_scale = self.config.get('texture_scale', 1.0)

		settings = MeshRendererSettings(enable_shadow=enable_shadow,
										enable_pbr=enable_pbr,
										msaa=False,
										texture_scale=texture_scale)

		self.simulator = Simulator(mode=mode,
								   physics_timestep=physics_timestep,
								   render_timestep=action_timestep,
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
								   external_camera_view_direction=self.config.get('external_camera_view_direction', [1, 0, 0]))

		self.load()								 

		self.automatic_reset = automatic_reset

	def load_scene_robot(self):
		"""
		Import the scene and robot (but have not reset their poses)
		"""
		if self.config['scene'] == 'relocate':
			scene_id = self.config['scene_id']
			n_interactive_objects = self.config.get('obj_num', 1)
			scene = RelocateScene(scene_id=scene_id, fix_interactive_objects=False, n_interactive_objects=n_interactive_objects)
			self.simulator.import_scene(scene, load_texture=self.config.get('load_texture', True))
		else:
			raise Exception(
				'unknown scene type: {}'.format(self.config['scene']))

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

		self.scene = scene
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
		drop_distance = 0.5 * 9.8 * (self.action_timestep ** 2)
		assert drop_distance < self.initial_pos_z_offset, \
			'initial_pos_z_offset is too small for collision checking'

		# ignore the agent's collision with these body ids
		self.collision_ignore_body_b_ids = set(
			self.config.get('collision_ignore_body_b_ids', []))
		
		# ignore the agent's collision with these link ids of itself
		self.collision_ignore_link_a_ids = set(
			self.config.get('collision_ignore_link_a_ids', []))

		# task
		if self.config['task'] == 'relocate_point_goal_fixed':
			self.task = RelocatePointGoalFixedTask(self)
		else:
			self.task = None
			print("No such task defined")	


	def load_observation_space(self):
		"""
		Load observation space
		"""
		self.output = self.config['output']
		self.image_width = self.config.get('image_width', 128)
		self.image_height = self.config.get('image_height', 128)
		observation_space = OrderedDict()
		sensors = OrderedDict()
		vision_modalities = []

		if 'task_obs' in self.output:
			observation_space['task_obs'] = self.build_obs_space(
				shape=(self.task.task_obs_dim,), low=-np.inf, high=-np.inf)
		if 'rgb' in self.output:
			observation_space['rgb'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 3),
				low=0.0, high=1.0)
			vision_modalities.append('rgb')    
		
		if len(vision_modalities) > 0:
			
			third_person_view = self.config.get("third_person_view", True)
			# third person view
			if third_person_view:
				sensors['vision'] = ExternalVisionSensor(self, vision_modalities, camera_pos=self.config.get('external_camera_pos', [0, 0, 1.2]),
								   camera_view_direction=self.config.get('external_camera_view_direction', [1, 0, 0]))
			# first person view
			else:
				sensors['vision'] = VisionSensor(self, vision_modalities)	

		self.observation_space = gym.spaces.Dict(observation_space)
		self.sensors = sensors

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
		self.current_step = 0
		self.non_interactive_collision_step = 0
		self.interactive_collision_step = 0
		self.current_episode = 0
		self.non_interactive_collision_links = []
		self.interactive_collision_links = []

	def load(self):
		"""
		Load environment
		"""
		self.load_scene_robot()  # load robot and scene, use self load()
		self.load_task_setup()
		self.load_observation_space()
		self.load_action_space()
		self.load_miscellaneous_variables()

	def load_observation_space(self):
		"""
		Load observation space
		"""
		self.output = self.config['output']
		self.image_width = self.config.get('image_width', 128)
		self.image_height = self.config.get('image_height', 128)
		observation_space = OrderedDict()
		sensors = OrderedDict()
		vision_modalities = []
		scan_modalities = []

		if 'task_obs' in self.output:
			observation_space['task_obs'] = self.build_obs_space(
				shape=(self.task.task_obs_dim+self.task.obj_num*12,), low=-np.inf, high=-np.inf)
		if 'rgb' in self.output:
			observation_space['rgb'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 3),
				low=0.0, high=1.0)
			vision_modalities.append('rgb')
		if 'depth' in self.output:
			observation_space['depth'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 1),
				low=0.0, high=1.0)
			vision_modalities.append('depth')
		if 'pc' in self.output:
			observation_space['pc'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 3),
				low=-np.inf, high=np.inf)
			vision_modalities.append('pc')
		if 'optical_flow' in self.output:
			observation_space['optical_flow'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 2),
				low=-np.inf, high=np.inf)
			vision_modalities.append('optical_flow')
		if 'scene_flow' in self.output:
			observation_space['scene_flow'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 3),
				low=-np.inf, high=np.inf)
			vision_modalities.append('scene_flow')
		if 'normal' in self.output:
			observation_space['normal'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 3),
				low=-np.inf, high=np.inf)
			vision_modalities.append('normal')
		if 'seg' in self.output:
			observation_space['seg'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 1),
				low=0.0, high=1.0)
			vision_modalities.append('seg')
		if 'rgb_filled' in self.output:  # use filler
			observation_space['rgb_filled'] = self.build_obs_space(
				shape=(self.image_height, self.image_width, 3),
				low=0.0, high=1.0)
			vision_modalities.append('rgb_filled')
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
			self.occupancy_grid_space = gym.spaces.Box(low=0.0,
													   high=1.0,
													   shape=(self.grid_resolution,
															  self.grid_resolution, 1))
			observation_space['occupancy_grid'] = self.occupancy_grid_space
			scan_modalities.append('occupancy_grid')

		if len(vision_modalities) > 0:
			sensors['vision'] = VisionSensor(self, vision_modalities)

		if len(scan_modalities) > 0:
			sensors['scan_occ'] = ScanSensor(self, scan_modalities)

		self.observation_space = gym.spaces.Dict(observation_space)	
		self.sensors = sensors

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
				state[modality] = vision_obs[modality]

		return state

	def run_simulation(self):
		"""
		Run simulation for one action timestep (same as one render timestep in Simulator class)

		:return: collision_links: collisions from last physics timestep
		"""
		self.simulator_step()
		collision_links = list(p.getContactPoints(bodyA=self.robots[0].robot_ids[0]))
		return self.filter_collision_links(collision_links)

	# get collision links with robot base link, ignore some, return collisions with interactive and non-interactive links respectively
	def filter_collision_links(self, collision_links):
		"""
		Filter out collisions that should be ignored

		:param collision_links: original collisions, a list of collisions
		:return: filtered collisions
		"""
		non_interactive_collision_links = []
		interactive_collision_links = []

		for item in collision_links:
			# ignore collision with robot link a
			if item[3] in self.collision_ignore_link_a_ids:
				continue

			# ignore self collision with robot link a (body b is also robot itself)
			if item[2] == self.robots[0].robot_ids[0] and item[4] in self.collision_ignore_link_a_ids:
				continue

			# ignore collision with body b - interactive objects
			if item[2] in self.collision_ignore_body_b_ids:
				interactive_collision_links.append(item)
			else:
				non_interactive_collision_links.append(item)

		return non_interactive_collision_links, interactive_collision_links

	# populate information into info
	def populate_info(self, info):
		"""
		Populate info dictionary with any useful information
		"""
		info['episode_length'] = self.current_step
		info['non_interactive_collision_step'] = self.non_interactive_collision_step # how many steps involve collision with non-interactive objects
		info['interactive_collision_step'] = self.interactive_collision_step # how many steps involve collision with interactive objects

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
		non_interactive_collision_links, interactive_collision_links = self.run_simulation()
		self.non_interactive_collision_links = non_interactive_collision_links
		self.interactive_collision_links = interactive_collision_links

		self.non_interactive_collision_step += int(len(non_interactive_collision_links) > 0)
		self.interactive_collision_step += int(len(interactive_collision_links) > 0)

		state = self.get_state()
		info = {}

		reward, done, info, sub_reward = self.task.get_reward_termination(self, info)

		print(sub_reward)

		# step task related variables
		self.task.step(self)

		self.populate_info(info)

		if done and self.automatic_reset:
			info['last_observation'] = state
			state = self.reset()

		return state, reward, done, info

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

	def reset_variables(self):
		"""
		Reset bookkeeping variables for the next new episode
		"""
		self.current_episode += 1
		self.current_step = 0
		self.non_interactive_collision_step = 0
		self.interactive_collision_step = 0
		self.non_interactive_collision_links = []
		self.interactive_collision_links = []


	def reset(self):
		"""
		Reset episode
		"""
		# move robot away from the scene
		self.robots[0].set_position([100.0, 100.0, 100.0])
		# reset scene
		self.task.reset_scene(self)
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
		help='which config file to use [default: use yaml files in examples/configs]', default=os.path.join(config_path,'turtlebot_relocate.yaml'))
	parser.add_argument('--mode',
						'-m',
						choices=['headless', 'gui', 'iggui'],
						default='headless',
						help='which mode for simulation (default: headless)')
	args = parser.parse_args()



	env = RelocateEnv(config_file=args.config,
					 mode=args.mode,
					 action_timestep=1.0 / 10.0,
					 physics_timestep=1.0 / 40.0)



	step_time_list = []
	for episode in range(100):
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
			print('-----------------------------')
			#print('-------------------------------')
			#print('reward', reward)
			#print(state['task_obs'].shape)
			if done:
				break
		
		print('Episode finished after {} timesteps, took {} seconds.'.format(
			env.current_step, time.time() - start))
	env.close()
