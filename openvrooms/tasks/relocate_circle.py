from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene

from openvrooms.reward_termination_functions.collision import PosNegCollision
from openvrooms.reward_termination_functions.timeout import Timeout

from openvrooms.reward_termination_functions.out_of_bound import OutOfBound
from openvrooms.reward_termination_functions.circle_goal import CircleGoal

from gibson2.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject

from openvrooms.scenes.relocate_scene import RelocateScene
from openvrooms.objects.interactive_object import InteractiveObj

import numpy as np

from gibson2.utils.utils import quatToXYZW, quatFromXYZW
from transforms3d.euler import euler2quat, quat2euler

from openvrooms.utils.utils import *

from openvrooms.tasks.relocate_goal_fixed_task import RelocateGoalFixedTask

class RelocateCircleTask(RelocateGoalFixedTask):
	"""
	Relocate Object Goal Fixed Task
	The goal is to push objects to fixed goal locations
	"""

	def __init__(self, env):
		super(RelocateCircleTask, self).__init__(env)

		self.reward_termination_functions = [
			Timeout(self.config),
			CircleGoal(self.config),
			PosNegCollision(self.config)
		]


		# get robot's initial pose
		self.agent_initial_pos = np.array(self.config.get('agent_initial_pos', [0, 0, 0]))
		self.agent_initial_orn = np.array(self.config.get('agent_initial_orn', [0, 0, 0]))  # euler angles: rotatation around x,y,z axis

		# get object intial positions (for scene and visualization)
		self.obj_initial_pos = np.array(self.config.get('obj_initial_pos'))
		self.obj_initial_orn = np.array(self.config.get('obj_initial_orn'))
		self.circle_radius = np.array(self.config.get('circle_radius'))

		self.obj_num = self.config.get('obj_num', 1)

		if self.obj_initial_pos.shape[0] != self.obj_num:
			raise Exception("Initial position list should have %d objects, instead of %d !"%(self.obj_num, self.obj_initial_pos.shape[0]))

		self.circle_num = self.circle_radius.shape[0]
		if self.obj_num == 1:
			assert self.circle_num == 1
		elif self.obj_num > 1:
			assert self.circle_num == self.obj_num - 1
		else:
			raise Exception("obj num should > 0")		

		print("Number of objects: %d"%(self.obj_num))
		print("Initial x-y positions of objects: \n%s"%self.obj_initial_pos)
		print("Circle radius: \n%s"%self.circle_radius)


		self.goal_format = self.config.get('goal_format', 'cartesian')

		self.visual_object_visible_to_agent = self.config.get(
			'visual_object_visible_to_agent', False
		)
		
		self.third_person_view = self.config.get("third_person_view", True)

		self.load_visualization(env)

		self.get_loaded_interactive_objects(env)

		# ignore collisions with interactive objects
		env.collision_ignore_body_b_ids |= set(
			[obj.body_id for obj in self.interactive_objects])

		
		# check validity of initial and target scene
		print("--------------- Check validity of initial scene ------------")
		self.check_initial_scene_collision(env)
		print("--------------------------------------------- ")
		

	def load_visualization(self, env):
		"""
		Load visualization, such as initial and target position, shortest path, etc

		:param env: environment instance
		"""
		if env.mode != 'gui':
			return

		vis_radius = float(self.config.get('dist_tol'))

		self.initial_pos_vis_objs = []
		for i in list(np.arange(self.obj_num)):
			cyl_length = env.scene.interative_objects[i].box_height + 0.2

			self.initial_pos_vis_objs.append(VisualMarker(
				visual_shape=p.GEOM_CYLINDER,
				rgba_color=[1, 0, 0, 0.3],
				radius=vis_radius,
				length=cyl_length,
				initial_offset=[0, 0, cyl_length / 2.0]))

		self.vis_circles = []
		for i in list(np.arrange(self.circle_num)):
			self.vis_circles.append(VisualMarker(
				visual_shape=p.GEOM_CYLINDER,
				rgba_color=[0, 0, 1, 0.3],
				radius=self.circle_radius[i],
				length=env.scene.interative_objects[0].box_height + 0.2,
				initial_offset=[0, 0, cyl_length / 2.0]))

			

		if self.visual_object_visible_to_agent:
			for i in list(np.arange(self.obj_num)):
				env.simulator.import_object(self.initial_pos_vis_objs[i])
			
			for i in list(np.arrange(self.circle_num)):
				env.simulator.import_object(self.vis_circles[i])
		else:
			for i in list(np.arange(self.obj_num)):
				self.initial_pos_vis_objs[i].load()
			
			for i in list(np.arrange(self.circle_num)):
				self.vis_circles[i].load()
	
	def circle_goal_distance(self):
		current_pos = self.get_obj_current_pos()
		# current_orn: n*4, quaterion [x,y,z,w]
		current_orn = self.get_obj_current_rot()
		goal_pos = self.get_obj_goal_pos()
		# goal_orn: n*3, euler angles
		goal_orn = self.get_obj_goal_rot()

		# current_orn, goal_orn: n*4, quaterion [w,x,y,z]
		goal_orn = euler2quat_array(goal_orn)
		current_orn = quatFromXYZW_array(current_orn, 'wxyz')
		#print(quatFromXYZW_array(quatToXYZW_array(goal_orn, 'wxyz'), 'wxyz'))

		# 1-1 mapping between objects and goals
		if self.duplicated_objects == False or self.obj_num == 1:
			# All the objects are different
			relative_obj_pos = goal_pos - current_pos
			relative_obj_rot = self.rot_dist_func(goal_orn, current_orn, output_angle=output_angle)
		# need to assign objects to closest goals
		else:
			assert current_pos.shape == goal_pos.shape
			assert current_orn.shape[0] == goal_orn.shape[0]
			# n*1*3, 1*n*3 --> n*n
			dist = np.linalg.norm(np.expand_dims(current_pos, 1) - np.expand_dims(goal_pos, 0), axis=-1)
			assert dist.shape == (self.obj_num, self.obj_num)

			relative_obj_pos = np.zeros([self.obj_num, goal_pos.shape[-1]]) # n*3
			relative_obj_rot = np.zeros([self.obj_num, goal_orn.shape[-1]]) # n*3
			
			for _ in range(self.obj_num):
				i, j = np.unravel_index(np.argmin(dist, axis=None), dist.shape)
				relative_obj_pos[i] = goal_pos[j] - current_pos[i]
				relative_obj_rot[i] = self.rot_dist_func(goal_orn[j], current_orn[i], output_angle=output_angle)
				# once we select a pair of match (i, j), wipe out their distance info.
				dist[i, :] = np.inf
				dist[:, j] = np.inf
			   
		# normalize angles to [-pi, pi] 
		relative_obj_rot = normalize_angles(relative_obj_rot)

		return relative_obj_pos, relative_obj_rot

	# for single object
	def get_reward_termination(self, env, info):
		"""
		Aggreate reward functions and episode termination conditions

		:param env: environment instance
		:return reward: total reward of the current timestep
		:return done: whether the episode is done
		:return info: additional info
		"""

		assert self.reward_termination_functions[0].get_name() == "timeout" 
		assert self.reward_termination_functions[1].get_name() == "circle_goal" 
		assert self.reward_termination_functions[2].get_name() == "negative_and_positive_collision" 

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
			# positive collision (push)	
			elif self.reward_termination_functions[2].has_positive_collision():	
				if self.config["use_goal_dist_reward"]:
					reward = float(self.config["collision_reward"]) + self.reward_termination_functions[1].goal_dist_reward
					#print(self.reward_termination_functions[1].goal_dist_reward)
				else:
					reward = float(self.config["collision_reward"])
			# time elapse (pure locomotion)
			else:
				reward = float(self.config["time_elapse_reward"])	

		return reward, done, info, sub_reward

	# for different objects
	def get_reward_termination_different_objects(self, env, info):
		"""
		Aggreate reward functions and episode termination conditions

		:param env: environment instance
		:return reward: total reward of the current timestep
		:return done: whether the episode is done
		:return info: additional info
		"""

		assert self.reward_termination_functions[0].get_name() == "timeout" 
		assert self.reward_termination_functions[1].get_name() == "circle_goal" 
		assert self.reward_termination_functions[2].get_name() == "negative_and_positive_collision" 

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

		# get tier 
		reward_tier = self.reward_termination_functions[1].get_reward_tier()

		# compute reward
		# succeed
		if self.reward_termination_functions[1].goal_reached():
			assert info['success'] == True
			reward = float(self.config["success_reward"])
		# not succeed	
		else:	
			# negative collision
			if self.reward_termination_functions[2].has_negative_collision():
				reward = float(self.config["collision_penalty"])
			# two tiers:	
			# positive collision	
			elif self.reward_termination_functions[2].has_positive_collision():	
				reward = float(self.config["collision_reward"]) + float(self.config["tier_cost"]) * reward_tier
			# time elapse
			else:
				reward = float(self.config["time_elapse_reward"]) + float(self.config["tier_cost"])* reward_tier	
				
		return reward, done, info, sub_reward		
 

	# visualize initial and target positions of the objects
	def step_visualization(self, env):
		"""
		Step visualization

		:param env: environment instance
		"""

		if env.mode != 'gui':
			return

		for i in list(np.arange(self.obj_num)):
			self.initial_pos_vis_objs[i].set_position([self.obj_initial_pos[i][0], self.obj_initial_pos[i][1], 0])
		
		for i in list(np.arrange(self.circle_num)):
			self.vis_circles[i].set_position([self.agent_initial_pos[i][0], self.agent_initial_pos[i][1], 0])

