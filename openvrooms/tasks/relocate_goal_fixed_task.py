from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene

from openvrooms.reward_termination_functions.collision import PosNegCollision
from openvrooms.reward_termination_functions.timeout import Timeout

from openvrooms.reward_termination_functions.out_of_bound import OutOfBound
from openvrooms.reward_termination_functions.object_goal import ObjectGoal

from gibson2.utils.utils import l2_distance, rotate_vector_3d, cartesian_to_polar
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject

from openvrooms.scenes.relocate_scene import RelocateScene
from openvrooms.objects.interactive_object import InteractiveObj

import numpy as np

from gibson2.utils.utils import quatToXYZW, quatFromXYZW
from transforms3d.euler import euler2quat, quat2euler

from openvrooms.utils.utils import *

class RelocateGoalFixedTask(BaseTask):
	"""
	Relocate Object Goal Fixed Task
	The goal is to push objects to fixed goal locations
	"""

	def __init__(self, env):
		super(RelocateGoalFixedTask, self).__init__(env)

		self.reward_termination_functions = [
			Timeout(self.config),
			OutOfBound(self.config, env),
			ObjectGoal(self.config),
			PosNegCollision(self.config)
		]


		# get robot's initial pose
		self.agent_initial_pos = np.array(self.config.get('agent_initial_pos', [0, 0, 0]))
		self.agent_initial_orn = np.array(self.config.get('agent_initial_orn', [0, 0, 0]))  # euler angles: rotatation around x,y,z axis

		# get object intial positions (for scene and visualization)
		self.obj_initial_pos = np.array(self.config.get('obj_initial_pos'))
		self.obj_initial_orn = np.array(self.config.get('obj_initial_orn'))
		self.obj_target_pos = np.array(self.config.get('obj_target_pos'))
		self.obj_target_orn = np.array(self.config.get('obj_target_orn'))


		if self.obj_initial_pos.shape[0] != self.obj_target_pos.shape[0]:
			raise Exception("Initial position list should have the same shape as target position list!")

		self.obj_num = self.config.get('obj_num', 1)
		

		if self.obj_initial_pos.shape[0] != self.obj_num:
			raise Exception("Initial position list should have %d objects, instead of %d !"%(self.obj_num, self.obj_initial_pos.shape[0]))


		print("Number of objects: %d"%(self.obj_num))
		print("Initial x-y positions of objects: \n%s"%self.obj_initial_pos)
		print("Target x-y positions of objects: \n%s"%self.obj_target_pos)

		self.duplicated_objects = self.config.get('duplicated_objects', True)

		self.goal_format = self.config.get('goal_format', 'cartesian')

		#self.visual_object_at_initial_target_pos = self.config.get(
		#    'visual_object_at_initial_target_pos', True
		#)
		self.visual_object_visible_to_agent = self.config.get(
			'visual_object_visible_to_agent', False
		)
		
		self.third_person_view = self.config.get("third_person_view", True)

		self.load_visualization(env)

		self.get_loaded_interactive_objects(env)

		# ignore collisions with interactive objects
		env.collision_ignore_body_b_ids |= set(
			[obj.body_id for obj in self.interactive_objects])

		#print(env.collision_ignore_body_b_ids)

		# check validity of initial and target scene
		print("--------------- Check validity of initial and target scene ------------")

		self.check_initial_scene_collision(env)
		self.check_target_scene_collision(env)
		print("--------------------------------------------- ")
		

	def get_loaded_interactive_objects(self, env):
		"""
		Get loaded interactive objects from the scene

		:param env: environment instance
		:return: a list of interactive objects
		"""
		self.interactive_objects = env.scene.interative_objects

		if len(self.interactive_objects) < 1:
			raise Exception("Interactive objects not loaded in the scene!")
		

	def reset_scene(self, env):
		"""
		Task-specific scene reset: reset the interactive objects after scene and agent reset

		:param env: environment instance
		"""
		env.scene.reset_interactive_object_poses(self.obj_initial_pos, self.obj_initial_orn)

	def load_visualization(self, env):
		"""
		Load visualization, such as initial and target position, shortest path, etc

		:param env: environment instance
		"""
		if env.mode != 'gui':
			return

		cyl_length = 0.2

		vis_radius = 0.2

		self.initial_pos_vis_objs = []
		for i in list(np.arange(self.obj_num)):
			self.initial_pos_vis_objs.append(VisualMarker(
				visual_shape=p.GEOM_CYLINDER,
				rgba_color=[1, 0, 0, 0.3],
				radius=vis_radius,
				length=cyl_length,
				initial_offset=[0, 0, cyl_length / 2.0]))

		self.target_pos_vis_objs = []
		for i in list(np.arange(self.obj_num)):
			self.target_pos_vis_objs.append(VisualMarker(
				visual_shape=p.GEOM_CYLINDER,
				rgba_color=[0, 0, 1, 0.3],
				radius=vis_radius,
				length=cyl_length,
				initial_offset=[0, 0, cyl_length / 2.0]))

		if self.visual_object_visible_to_agent:
			for i in list(np.arange(self.obj_num)):
				env.simulator.import_object(self.initial_pos_vis_objs[i])
				env.simulator.import_object(self.target_pos_vis_objs[i])
		else:
			for i in list(np.arange(self.obj_num)):
				self.initial_pos_vis_objs[i].load()
				self.target_pos_vis_objs[i].load()

	
	# goal_orn, current_orn: [w,x,y,z]
	# output: angle distance or quaternion distance
	def rot_dist_func(self, goal_orn, current_orn, output_angle=True):
		#return subtract_euler(goal_state["obj_rot"], current_state["obj_rot"])
		q_diff = subtract_quat(goal_orn, current_orn)

		if output_angle:
			return quat2euler_array(q_diff)
		else:
			return q_diff    

	def relative_goal(self, output_angle=True):
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

		if self.duplicated_objects == False or self.obj_num == 1:
			# All the objects are different
			relative_obj_pos = goal_pos - current_pos
			relative_obj_rot = self.rot_dist_func(goal_orn, current_orn, output_angle=output_angle)

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


	def goal_distance(self):
		relative_pos, relative_orn = self.relative_goal(output_angle=False)
		pos_distances = np.linalg.norm(relative_pos, axis=-1)
		rot_distances = quat_magnitude(quat_normalize(relative_orn))
		
		return pos_distances, rot_distances

	def check_initial_scene_collision(self, env):
		state_id = p.saveState()

		success = env.test_valid_position(env.robots[0],  self.agent_initial_pos,  self.agent_initial_orn)
		if not success:
			print("Initial scene Failed: unable to set robot initial pose without collision.")
		
		for i, obj in enumerate(env.scene.interative_objects):    
			success = env.test_valid_position(obj,  [self.obj_initial_pos[i][0], self.obj_initial_pos[i][1], obj.goal_z],  self.obj_initial_orn[i])
			if not success:
				print("Initial scene Failed: unable to set object %d's initial pose without collision."%(i))
				

		p.restoreState(state_id)
		p.removeState(state_id)
		print("Validity check of initial scene Finished!")

	def check_target_scene_collision(self, env):
		state_id = p.saveState()
		
		for i, obj in enumerate(env.scene.interative_objects):    
			success = env.test_valid_position(obj,  [self.obj_target_pos[i][0], self.obj_target_pos[i][1], obj.goal_z],  self.obj_target_orn[i])
			if not success:
				print("Target scene Failed: unable to set object %d's target pose without collision."%(i))

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
		robot_linear_velocity = agent.get_linear_velocity()
		
		# 3d in world frame if third person view is adopted
		robot_angular_velocity = agent.get_angular_velocity()
		

		# 12 d in total
		task_obs = np.concatenate((robot_position, robot_orientation, robot_linear_velocity, robot_angular_velocity), axis=None)
		
		
		# object current pose: 6d each
		for obj in self.interactive_objects:
			pos, orn = obj.get_position_orientation()

			orn = np.array(orn)
			orn = quat2euler(quatFromXYZW(orn, 'wxyz'))

			task_obs = np.append(task_obs, pos)
			task_obs = np.append(task_obs, orn)

		
		# object target pose: 6d each
		for i, obj in enumerate(self.interactive_objects):
			target_pos = [self.obj_target_pos[i][0], self.obj_target_pos[i][1], obj.goal_z]
			task_obs = np.append(task_obs, target_pos)

			orn = np.array(self.obj_target_orn[i])
			task_obs = np.append(task_obs, orn)

		#print(task_obs.shape)
		return task_obs

	# n*3
	def get_obj_current_pos(self):
		pos_array = []
		for obj in self.interactive_objects:
			pos, _ = obj.get_position_orientation() 
			pos_array.append(pos)
		
		pos_array = np.array(pos_array)

		return pos_array
		
	# n*4, quaternion [x,y,z,w]     
	def get_obj_current_rot(self):
		rot_array = []
		for obj in self.interactive_objects:
			_, rot = obj.get_position_orientation() 
			rot_array.append(rot)
		
		rot_array = np.array(rot_array)

		return rot_array

	# n*3
	def get_obj_goal_pos(self):
		pos_array = []
		
		for i, obj in enumerate(self.interactive_objects):
			pos = [self.obj_target_pos[i][0], self.obj_target_pos[i][1], obj.goal_z] 
			pos_array.append(pos)
		
		pos_array = np.array(pos_array)

		return pos_array  

	# n*3, euler angle
	def get_obj_goal_rot(self):
		rot_array = []
		
		for i in list(range(self.obj_num)):
			rot_array.append(self.obj_target_orn[i])
		
		rot_array = np.array(rot_array)

		return rot_array      

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
			self.target_pos_vis_objs[i].set_position([self.obj_target_pos[i][0], self.obj_target_pos[i][1], 0])


	def step(self, env):
		"""
		Perform task-specific step: step visualization and aggregate path length

		:param env: environment instance
		"""
		self.step_visualization(env)
		new_robot_pos = env.robots[0].get_position()[:2]
		self.robot_pos = new_robot_pos
