from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene

from openvrooms.reward_termination_functions.collision import PosNegCollision
from openvrooms.reward_termination_functions.timeout import Timeout

from openvrooms.reward_termination_functions.out_of_bound import OutOfBound
from openvrooms.reward_termination_functions.region_goal import RegionGoal

from gibson2.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject

from openvrooms.scenes.relocate_scene import RelocateScene
from openvrooms.objects.interactive_object import InteractiveObj

import numpy as np
import math
import random
import copy

from gibson2.utils.utils import quatToXYZW, quatFromXYZW
from transforms3d.euler import euler2quat, quat2euler

from openvrooms.utils.utils import *
from openvrooms.config import *

from openvrooms.tasks.relocate_goal_fixed_task import RelocateGoalFixedTask

class RelocateRegionTask(RelocateGoalFixedTask):
	"""
	Relocate Object Goal Fixed Task
	The goal is to push objects to fixed goal locations
	"""

	def __init__(self, env):
		super(RelocateGoalFixedTask, self).__init__(env)

		self.reward_termination_functions = [
			Timeout(self.config),
			RegionGoal(self.config),
			PosNegCollision(self.config)
		]


		# get robot's initial pose
		self.agent_initial_pos = np.array(self.config.get('agent_initial_pos', [0, 0, 0]))
		self.agent_initial_orn = np.array(self.config.get('agent_initial_orn', [0, 0, 0]))  # euler angles: rotatation around x,y,z axis

		# get object intial positions (for scene and visualization)
		self.obj_initial_pos = np.array(self.config.get('obj_initial_pos'))
		self.obj_initial_orn = np.array(self.config.get('obj_initial_orn'))
		self.region_boundary = np.array(self.config.get('region_boundary'))

		self.obj_num = self.config.get('obj_num', 1)

		if self.obj_initial_pos.shape[0] != self.obj_num:
			raise Exception("Initial position list should have %d objects, instead of %d !"%(self.obj_num, self.obj_initial_pos.shape[0]))
				

		print("Number of objects: %d"%(self.obj_num))
		print("Initial x-y positions of objects: \n%s"%self.obj_initial_pos)
		print("Region boundary: \n%s"%self.region_boundary)

		#self.sparser_reward = self.config.get("0_1_reward", False)
		self.reward_function_choice = self.config.get("reward_function_choice", "0-1-push-time")

		self.y_flip = self.config.get("y_flip", False)

		self.goal_conditioned = False

		self.goal_format = self.config.get('goal_format', 'cartesian')

		self.visual_object_visible_to_agent = self.config.get(
			'visual_object_visible_to_agent', False
		)

		self.visual_object_visible_to_human = self.config.get(
			'visual_object_visible_to_human', False
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
		for i in list(range(self.obj_num)):
			cyl_length = env.scene.interative_objects[i].box_height + 0.2

			self.initial_pos_vis_objs.append(VisualMarker(
				visual_shape=p.GEOM_CYLINDER,
				rgba_color=[1, 0, 0, 0.3],
				radius=vis_radius,
				length=cyl_length,
				initial_offset=[0, 0, cyl_length / 2.0]))

		
		self.vis_region_x = [self.region_boundary[0], (env.scene.x_range[1]-env.scene.x_range[0])/2.0]
		if self.region_boundary.size < 2:
			self.vis_region_y = [-(env.scene.y_range[1]-env.scene.y_range[0])/2.0, (env.scene.y_range[1]-env.scene.y_range[0])/2.0]
		else:
			if self.y_flip == False:
				self.vis_region_y = [self.region_boundary[1], (env.scene.y_range[1]-env.scene.y_range[0])/2.0]
			else:
				self.vis_region_y = [-(env.scene.y_range[1]-env.scene.y_range[0])/2.0, self.region_boundary[1]]	
		

		vis_region_x_half_extent = (self.vis_region_x[1] - self.vis_region_x[0]) / 2.0
		vis_region_y_half_extent = (self.vis_region_y[1] - self.vis_region_y[0]) / 2.0
		self.vis_region_offset = [self.vis_region_x[0]+vis_region_x_half_extent, self.vis_region_y[0]+vis_region_y_half_extent, 0]

		self.vis_region = VisualMarker(
			visual_shape=p.GEOM_BOX,
			rgba_color=[0, 1, 0, 0.3],
			half_extents=[vis_region_x_half_extent, vis_region_y_half_extent, 0.1],
			initial_offset=[0, 0, 0.1])

		if self.visual_object_visible_to_agent:
			for i in list(range(self.obj_num)):
				env.simulator.import_object(self.initial_pos_vis_objs[i])
			
			env.simulator.import_object(self.vis_region)
		else:
			for i in list(range(self.obj_num)):
				self.initial_pos_vis_objs[i].load()

			self.vis_region.load()
	
	def region_goal_distance(self):
		# pose
		current_pos = self.get_obj_current_pos()
		# position x,y
		current_pos = current_pos[:,:2]
		
		assert current_pos.shape[0] == 1
		# x upper boundary
		region_distance = self.region_boundary[0] - current_pos[0][0] 

		return region_distance

	def get_current_xy_position(self):	
		# pose
		current_pos = self.get_obj_current_pos()
		# position x,y
		current_pos = current_pos[0,:2]

		return current_pos

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
		assert self.reward_termination_functions[1].get_name() == "region_goal" 
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
		# 0-1 reward structure
		if self.reward_function_choice == "0-1-push-time": # in use
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
					reward = float(self.config["collision_reward"])
				# time elapse (pure locomotion)
				else:
					reward = float(self.config["time_elapse_reward"])	
		elif self.reward_function_choice == "0-1-push-time-with-energy": 
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
					ratio = env.compute_step_energy_ratio()
					#print(ratio)
					reward = float(self.config["collision_reward"]) * float(ratio)
				#elif self.reward_termination_functions[2].has_positive_collision():	
				#	obj = env.scene.interative_objects[0]
				#	floor_friction_coefficient, object_mass, current_pos_xy, current_orn_z, obj_x_width, obj_y_width = env.get_interactive_obj_physics(obj)
				#	print("fc: %f"%(floor_friction_coefficient))
					#reward = float(self.config["collision_reward"])
				#	reward = float(self.config["time_elapse_reward"]) * floor_friction_coefficient
				# time elapse (pure locomotion)
				else:
					reward = float(self.config["time_elapse_reward"])			
		elif self.reward_function_choice == "0-1": # in use, use episode energy if consider energy
			# goal reached
			if self.reward_termination_functions[1].goal_reached():
				assert info['success'] == True
				reward = float(self.config["success_reward"])
			# not succeed	
			else:	
				# negative collision
				if self.reward_termination_functions[2].has_negative_collision():
					reward = float(self.config["collision_penalty"])
				else:
					reward = 0.0			
		elif self.reward_function_choice == "0-1-time":
			# goal reached
			if self.reward_termination_functions[1].goal_reached():
				assert info['success'] == True
				reward = float(self.config["success_reward"])
			# not succeed	
			else:	
				# negative collision
				if self.reward_termination_functions[2].has_negative_collision():
					reward = float(self.config["collision_penalty"])
				# time elapse (do not distinguish pure locomotion and pushing)
				else:
					reward = float(self.config["time_elapse_reward"])			
		# -1-0 reward structure			
		elif self.reward_function_choice == "-1-0-time":
			# goal reached
			if self.reward_termination_functions[1].goal_reached():
				assert info['success'] == True
				reward = 0.0
			# not succeed	
			else:	
				# negative collision
				if self.reward_termination_functions[2].has_negative_collision():
					reward = float(self.config["collision_penalty"])
				# time elapse (do not distinguish pure locomotion and pushing)
				else:
					reward = float(self.config["time_elapse_reward"])
		elif self.reward_function_choice == "-1-0-push-time": # in use
			# goal reached
			if self.reward_termination_functions[1].goal_reached():
				assert info['success'] == True
				reward = 0.0
			# not succeed	
			else:	
				# negative collision
				if self.reward_termination_functions[2].has_negative_collision():
					reward = float(self.config["collision_penalty"])
				# positive collision (push)	
				elif self.reward_termination_functions[2].has_positive_collision():	
					reward = float(self.config["collision_reward"])
				# time elapse (pure locomotion)
				else:
					reward = float(self.config["time_elapse_reward"])	
		elif self.reward_function_choice == "-1-0-push-time-with-energy":   # in use
			# goal reached
			if self.reward_termination_functions[1].goal_reached():
				assert info['success'] == True
				reward = 0.0
			# not succeed	
			else:	
				# negative collision
				if self.reward_termination_functions[2].has_negative_collision():
					reward = float(self.config["collision_penalty"])
				# positive collision (push)	
				elif self.reward_termination_functions[2].has_positive_collision():	
					ratio = env.compute_step_energy_ratio()
					#print(ratio)
					reward = float(self.config["collision_reward"]) * float(ratio)
				# time elapse (pure locomotion)
				else:
					reward = float(self.config["time_elapse_reward"])															
		else:
			print("Error: unknown reward function type")			

		return reward, done, info, sub_reward

	# visualize initial and target positions of the objects
	def step_visualization(self, env):
		"""
		Step visualization

		:param env: environment instance
		"""

		if env.mode != 'gui':
			return

		
		for i in list(range(self.obj_num)):
				self.initial_pos_vis_objs[i].set_position([self.obj_initial_pos[i][0], self.obj_initial_pos[i][1], 0])
		
		
		self.vis_region.set_position(self.vis_region_offset)


	def recover_pybullet(self, state_id):
		p.restoreState(state_id)
		p.removeState(state_id)

	def check_initial_pose(self, env, robot_orn, obj_init_pose):
		state_id = p.saveState()

		success = env.test_valid_position(env.robots[0],  self.agent_initial_pos,  [0,0,robot_orn])
		if not success:
			self.recover_pybullet(state_id)
			return False
		
		for i, obj in enumerate(env.scene.interative_objects):    
			success = env.test_valid_position(obj,  [obj_init_pose[i][0], obj_init_pose[i][1], obj.goal_z],  [0,0,obj_init_pose[i][2]])
			if not success:
				self.recover_pybullet(state_id)
				return False

		self.recover_pybullet(state_id)
		return True

	


if __name__ == '__main__':
		print('Done')

