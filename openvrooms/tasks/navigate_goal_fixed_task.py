from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene

from openvrooms.reward_termination_functions.collision import NegCollision
from openvrooms.reward_termination_functions.timeout import Timeout

from openvrooms.reward_termination_functions.out_of_bound import OutOfBound
from openvrooms.reward_termination_functions.point_goal import PointGoal

from gibson2.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject

from openvrooms.scenes.relocate_scene import RelocateScene
from openvrooms.objects.interactive_object import InteractiveObj

import numpy as np

from gibson2.utils.utils import quatToXYZW, quatFromXYZW
from transforms3d.euler import euler2quat, quat2euler

from openvrooms.utils.utils import *

class NavigateGoalFixedTask(BaseTask):
	"""
	Navigate Point Goal Fixed Task
	The goal is reach a given goal locations
	"""

	def __init__(self, env):
		super(NavigateGoalFixedTask, self).__init__(env)

		self.reward_termination_functions = [
			Timeout(self.config),
			PointGoal(self.config),
			NegCollision(self.config),
			#OutOfBound(self.config, env),
		]


		# get robot's initial pose and target pose
		self.agent_initial_pos = np.array(self.config.get('agent_initial_pos', [0, 0, 0]))
		self.agent_initial_orn = np.array(self.config.get('agent_initial_orn', [0, 0, 0]))  # euler angles: rotatation around x,y,z axis
		self.agent_target_pos = np.array(self.config.get('agent_target_pos', [1, 1, 0])) 

		# get obstacle poses (for scene and visualization)
		self.obs_num = self.config.get('obs_num', 0)
		if self.obs_num > 0:
			self.obs_pos = np.array(self.config.get('obs_pos'))
			self.obs_orn = np.array(self.config.get('obs_orn'))

			if self.obs_pos.shape[0] != self.obs_num:
				raise Exception("Obstacle position list should have %d objects, instead of %d !"%(self.obs_num, self.obs_pos.shape[0]))


		print("Number of obstacles: %d"%(self.obs_num))

		if self.obs_num > 0:
			print("x-y positions of obstacles: \n%s"%self.obs_pos)
		
		self.goal_format = self.config.get('goal_format', 'cartesian')

		#self.visual_object_at_initial_target_pos = self.config.get(
		#    'visual_object_at_initial_target_pos', True
		#)
		self.visual_object_visible_to_agent = self.config.get(
			'visual_object_visible_to_agent', False
		)
		
		self.third_person_view = self.config.get("third_person_view", True)

		self.load_visualization(env)

		# load obstacles
		if self.obs_num > 0:
			self.get_loaded_obstacles(env)

		# no ignore collision objects
		env.collision_ignore_body_b_ids = []

		#print(env.collision_ignore_body_b_ids)

		# check validity of initial and target scene if there are obstacles
		if self.obs_num > 0:
			print("--------------- Check validity of initial and target position of robot ------------")
			self.check_initial_scene_collision(env)
			self.check_target_scene_collision(env)
			print("--------------------------------------------- ")

		# set fixed poses of obstacles
		if self.obs_num > 0:
			env.scene.reset_interactive_object_poses(self.obs_pos, self.obs_orn)
		

	def get_loaded_obstacles(self, env):
		"""
		Get loaded interactive objects from the scene

		:param env: environment instance
		:return: a list of interactive objects
		"""
		self.interactive_objects = env.scene.interative_objects

		if len(self.interactive_objects) < 1:
			raise Exception("Interactive objects not loaded in the scene!")
		

	def load_visualization(self, env):
		"""
		Load visualization, such as initial and target robot position, shortest path, etc

		:param env: environment instance
		"""
		if env.mode != 'gui':
			return

		cyl_length = 1.0

		#vis_radius = self.config.get('body_width', 0.4) / 2.0
		vis_radius = float(self.config.get('dist_tol'))

		self.initial_pos_vis_obj = VisualMarker(
				visual_shape=p.GEOM_CYLINDER,
				rgba_color=[1, 0, 0, 0.3],
				radius=vis_radius,
				length=cyl_length,
				initial_offset=[0, 0, cyl_length / 2.0])


		self.target_pos_vis_obj = VisualMarker(
				visual_shape=p.GEOM_CYLINDER,
				rgba_color=[0, 0, 1, 0.3],
				radius=vis_radius,
				length=cyl_length,
				initial_offset=[0, 0, cyl_length / 2.0])


		if self.visual_object_visible_to_agent:
			env.simulator.import_object(self.initial_pos_vis_obj)
			env.simulator.import_object(self.target_pos_vis_obj)
		else:
			self.initial_pos_vis_obj.load()
			self.target_pos_vis_obj.load()


	def goal_distance(self, env):
		agent_current_pos = env.robots[0].get_position()
		relative_pos = self.agent_target_pos - agent_current_pos
		pos_distance = np.linalg.norm(relative_pos, axis=-1)
		
		return pos_distance

	def check_initial_scene_collision(self, env):
		state_id = p.saveState()

		success = env.test_valid_position(env.robots[0],  self.agent_initial_pos,  self.agent_initial_orn)
		if not success:
			print("Initial scene Failed: unable to set robot initial pose without collision.")
		
		for i, obj in enumerate(env.scene.interative_objects):    
			success = env.test_valid_position(obj,  [self.obs_pos[i][0], self.obs_pos[i][1], obj.goal_z],  self.obs_orn[i])
			if not success:
				print("Initial scene Failed: unable to set obstacle %d's initial pose without collision."%(i))
				

		p.restoreState(state_id)
		p.removeState(state_id)
		print("Validity check of initial scene Finished!")

	def check_target_scene_collision(self, env):
		state_id = p.saveState()

		success = env.test_valid_position(env.robots[0],  self.agent_target_pos,  self.agent_initial_orn)
		if not success:
			print("Target scene Failed: unable to set robot target pose without collision.")

		for i, obj in enumerate(env.scene.interative_objects):    
			success = env.test_valid_position(obj,  [self.obs_pos[i][0], self.obs_pos[i][1], obj.goal_z],  self.obs_orn[i])
			if not success:
				print("Target scene Failed: unable to set obstacle %d's target pose without collision."%(i))
				

		p.restoreState(state_id)
		p.removeState(state_id)
		print("Validity check of target scene Finished!")	

	def reset_agent(self, env):
		"""
		Task-specific agent reset: land the robot to initial pose, compute initial potential

		:param env: environment instance
		"""

		# land robot at initial pose
		env.land(env.robots[0], self.agent_initial_pos, self.agent_initial_orn)
	
		# robot x,y
		self.robot_pos = self.agent_initial_pos[:2]

		# reset reward functions
		for reward_termination_function in self.reward_termination_functions:
			reward_termination_function.reset(self, env)

	'''
	def get_reward_termination(self, env, info):
		"""
		Aggreate reward functions and episode termination conditions

		:param env: environment instance
		:return reward: total reward of the current timestep
		:return done: whether the episode is done
		:return info: additional info
		"""

		reward = 0.0  # total reward
		done = False
		success = False

		sub_reward = {}

		for reward_termination in self.reward_termination_functions:
			r, d, s = reward_termination.get_reward_termination(self, env)

			reward += r
			done = done or d
			success = success or s

			sub_reward[reward_termination.get_name()] = r

		#info['done'] = done
		info['success'] = success

		return reward, done, info, sub_reward
	'''
	
	def get_reward_termination(self, env, info):
		"""
		Aggreate reward functions and episode termination conditions

		:param env: environment instance
		:return reward: total reward of the current timestep
		:return done: whether the episode is done
		:return info: additional info
		"""

		assert self.reward_termination_functions[0].get_name() == "timeout" 
		assert self.reward_termination_functions[1].get_name() == "point_goal" 
		assert self.reward_termination_functions[2].get_name() == "negative_collision" 

		# get done, success, and sub reward
		done = False
		success = False

		sub_reward = {}

		for reward_termination in self.reward_termination_functions:
			r, d, s = reward_termination.get_reward_termination(self, env)

			#reward += r
			done = done or d
			success = success or s

			sub_reward[reward_termination.get_name()] = r

		#info['done'] = done
		info['success'] = success

		# get reward
		# goal reached
		if self.reward_termination_functions[1].goal_reached():
			assert info['success'] == True
			reward = float(self.config["success_reward"])
		# not succeed	
		else:	
			# negative collision
			if self.reward_termination_functions[2].has_negative_collision():
				reward = float(self.config["collision_penalty"])
			# time elapse (pure locomotion)
			else:
				reward = float(self.config["time_elapse_reward"])	

		return reward, done, info, sub_reward

	# useful when the first person view is adopted
	def global_to_local(self, env, pos):
		"""
		Convert a 3D point in global frame to agent's local frame

		:param env: environment instance
		:param pos: a 3D point in global frame
		:return: the same 3D point in agent's local frame
		"""
		return rotate_vector_3d(pos - env.robots[0].get_position(),
								*env.robots[0].get_rpy())

	def get_task_obs(self, env):
		"""
		Get task-specific observation, including goal position, current velocities, etc.

		:param env: environment instance
		:return: task-specific observation
		"""

		agent = env.robots[0]

		# [x,y,z]
		robot_position = agent.get_position()
		
		# [x,y,z,w] --> [w,x,y,z] --> euler angles
		robot_orientation = np.array(agent.get_orientation())
		robot_orientation = quat2euler(quatFromXYZW(robot_orientation, 'wxyz'))

		
		# 3d in world frame if third person view is adopted
		#robot_linear_velocity = agent.get_linear_velocity()
		
		# 3d in world frame if third person view is adopted
		#robot_angular_velocity = agent.get_angular_velocity()


		# concatenated observations
		#task_obs = np.concatenate((robot_position, robot_orientation, robot_linear_velocity, robot_angular_velocity), axis=None)
		task_obs = np.concatenate((robot_position, robot_orientation), axis=None)
		#task_obs = robot_position

		#print(task_obs.shape)
		return task_obs
	

	# visualize initial and target positions of robots
	def step_visualization(self, env):
		"""
		Step visualization

		:param env: environment instance
		"""
		if env.mode != 'gui':
			return

		self.initial_pos_vis_obj.set_position(self.agent_initial_pos)
		self.target_pos_vis_obj.set_position(self.agent_target_pos)


	def step(self, env):
		"""
		Perform task-specific step: step visualization and aggregate path length

		:param env: environment instance
		"""
		self.step_visualization(env)
		new_robot_pos = env.robots[0].get_position()[:2]
		self.robot_pos = new_robot_pos
