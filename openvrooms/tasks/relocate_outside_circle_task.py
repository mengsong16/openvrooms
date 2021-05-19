from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene

from openvrooms.reward_termination_functions.collision import PosNegCollision
from openvrooms.reward_termination_functions.timeout import Timeout

from openvrooms.reward_termination_functions.out_of_bound import OutOfBound
from openvrooms.reward_termination_functions.outside_circle_goal import OutsideCircleGoal

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

class RelocateOutsideCircleTask(RelocateGoalFixedTask):
	"""
	Relocate Object Goal Fixed Task
	The goal is to push objects to fixed goal locations
	"""

	def __init__(self, env):
		super(RelocateGoalFixedTask, self).__init__(env)

		self.reward_termination_functions = [
			Timeout(self.config),
			OutsideCircleGoal(self.config),
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
		
		assert self.circle_num == self.obj_num - 1
				

		print("Number of objects: %d"%(self.obj_num))
		print("Initial x-y positions of objects: \n%s"%self.obj_initial_pos)
		print("Circle radius: \n%s"%self.circle_radius)

		self.sparser_reward = self.config.get("0_1_reward", True)

		if self.sparser_reward:
			print("Use 0-1 reward")
		else:
			print("Do NOT Use 0-1 reward")	


		self.goal_format = self.config.get('goal_format', 'cartesian')

		self.visual_object_visible_to_agent = self.config.get(
			'visual_object_visible_to_agent', False
		)

		self.visual_object_visible_to_human = self.config.get(
			'visual_object_visible_to_human', False
		)
		
		self.third_person_view = self.config.get("third_person_view", True)

		self.random_init_pose = self.config.get('random_init_pose', False)
		self.load_visualization(env)

		self.get_loaded_interactive_objects(env)

		# ignore collisions with interactive objects
		env.collision_ignore_body_b_ids |= set(
			[obj.body_id for obj in self.interactive_objects])

		# whether take object goal as states
		self.goal_conditioned = False

		self.swap = self.config.get('swap')
		self.config_index = int(self.config.get('config_index'))
		self.get_configurations()
		 
		
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

		self.vis_circles = []
		for i in list(range(self.circle_num)):
			if i % 2 == 0:
				color = [0, 0, 1, 0.3]
			else:
				color = [0, 1, 0, 0.3]	
			self.vis_circles.append(VisualMarker(
				visual_shape=p.GEOM_CYLINDER,
				rgba_color=color,
				radius=self.circle_radius[i],
				length=0.2,
				initial_offset=[0, 0, 0]))

			

		if self.visual_object_visible_to_agent:
			for i in list(range(self.obj_num)):
				env.simulator.import_object(self.initial_pos_vis_objs[i])
			
			for i in list(range(self.circle_num)):
				env.simulator.import_object(self.vis_circles[i])
		else:
			'''
			if not self.random_init_pose:
				for i in list(range(self.obj_num)):
					self.initial_pos_vis_objs[i].load()
			'''
			for i in list(range(self.circle_num)):
				self.vis_circles[i].load()
	
	def circle_goal_distance(self):
		current_pos = self.get_obj_current_pos()
		current_pos = current_pos[:,:2]
		circle_center = np.tile(np.array(self.agent_initial_pos[:2]), (self.obj_num,1))

		assert current_pos.shape == circle_center.shape
		pos_distance = current_pos - circle_center

		pos_distance = np.linalg.norm(pos_distance, axis=-1)

		return pos_distance

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
		assert self.reward_termination_functions[1].get_name() == "outside_circle_goal" 
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
		if self.sparser_reward == False:
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
		# 0-1 reward
		else:
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
		assert self.reward_termination_functions[1].get_name() == "outside_circle_goal" 
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

		'''
		if not self.random_init_pose:
			for i in list(range(self.obj_num)):
				self.initial_pos_vis_objs[i].set_position([self.obj_initial_pos[i][0], self.obj_initial_pos[i][1], 0])
		'''	
		for i in list(range(self.circle_num)):
			self.vis_circles[i].set_position([self.agent_initial_pos[0], self.agent_initial_pos[1], 0])

	def random_initial_pose(self, env):	
		R = self.circle_radius[0]
		obj_init_pose = []
		for i in range(len(env.scene.interative_objects)):
			x, y = self.random_point_in_circle(R)
			z_angle = self.random_orientation()
			obj_init_pose.append([x, y, z_angle])

		robot_orn = self.random_orientation()	
		return	robot_orn, obj_init_pose
	
	def random_point_in_circle(self, R, centerX=0.0, centerY=0.0):
		r = R * math.sqrt(np.random.uniform(low=0.0, high=1.0))
		theta = np.random.uniform(low=0.0, high=1.0) * 2.0 * math.pi

		x = centerX + r * math.cos(theta)
		y = centerY + r * math.sin(theta)

		return x, y

	def random_orientation(self):
		z_angle = np.random.uniform(low=-math.pi, high=math.pi)

		return z_angle

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

	def reset_scene_agent_random(self, env):
		'''
		correct_scene = False
		robot_orn = 0
		obj_init_pose = []

		while correct_scene == False:
			robot_orn, obj_init_pose = self.random_initial_pose(env)
			correct_scene = self.check_initial_pose(env, robot_orn, obj_init_pose)
		'''
		obj_init_pose, robot_orn = self.random_choose_predefined_pose()	

		obj_initial_pos = []
		obj_initial_orn = []
		for i in range(len(obj_init_pose)):
			obj_initial_pos.append([obj_init_pose[i][0], obj_init_pose[i][1]])
			obj_initial_orn.append([0,0,obj_init_pose[i][2]])

		env.scene.reset_interactive_object_poses(np.array(obj_initial_pos), np.array(obj_initial_orn))

		# land robot at initial pose
		env.land(env.robots[0], self.agent_initial_pos, [0,0,robot_orn])
	
		# robot x,y
		self.robot_pos = self.agent_initial_pos[:2]

		# reset reward functions
		for reward_termination_function in self.reward_termination_functions:
			reward_termination_function.reset(self, env)

	def get_configurations(self):
		obj_init_pose_base = [[[0.7, 0.7, 0], [0.7, -0.7, 0]], \
						[[-0.7, 0.7, -math.pi], [-0.7, -0.7, -math.pi]], \
						[[0.7, -0.7, math.pi/2.0], [-0.7, -0.7, math.pi/2.0]], \
						[[0.7, 0.7, -math.pi/2.0], [-0.7, 0.7, -math.pi/2.0]]]

		self.configurations = []
		for i in range(4):
			for j in range(2):
				if j == 0:
					current_config = copy.deepcopy(obj_init_pose_base)
				else:
					current_config = copy.deepcopy(obj_init_pose_base)
					current_config[i][0], current_config[i][1] = current_config[i][1], current_config[i][0]	

				self.configurations.append(current_config)



	def print_configurations(self):
		for i, con in enumerate(self.configurations):
			print("=================================")
			print("Config: %d"%(i))
			for facing_dir in con:
				print("---------------------------------")
				print(str(facing_dir))
			print("=================================")
			
	def random_choose_predefined_pose(self):
			
		configurations = self.get_configurations()
		
		robot_orn = [0, -math.pi, -math.pi/2.0, math.pi/2.0]

		index = random.choice(list(range(4)))
		#obj_pose = obj_init_pose[index]
		obj_pose = self.configurations[self.config_index][index]

		#swap = random.choice(list(range(2)))
		#swap = 1

		#if swap == 0:
		if self.swap == False:
			return obj_pose, robot_orn[index]
		# swap two box		
		else:
			obj_pose[0], obj_pose[1] = obj_pose[1], obj_pose[0]		
			return obj_pose, robot_orn[index]

'''
def get_configurations():
		obj_init_pose_base = [[[0.7, 0.7, 0], [0.7, -0.7, 0]], \
						[[-0.7, 0.7, -math.pi], [-0.7, -0.7, -math.pi]], \
						[[0.7, -0.7, math.pi/2.0], [-0.7, -0.7, math.pi/2.0]], \
						[[0.7, 0.7, -math.pi/2.0], [-0.7, 0.7, -math.pi/2.0]]]

		configurations = []
		for i in range(4):
			for j in range(2):
				if j == 0:
					current_config = copy.deepcopy(obj_init_pose_base)
				else:
					current_config = copy.deepcopy(obj_init_pose_base)
					current_config[i][0], current_config[i][1] = current_config[i][1], current_config[i][0]	

				configurations.append(current_config)

		return configurations

def print_configurations(configurations):
	with open('circle_configurations.txt', 'w') as f:
		for i, con in enumerate(configurations):
			f.write("=================================\n")
			f.write("Config: %d\n"%(i+1))
			for facing_dir in con:
				f.write("---------------------------------\n")
				f.write(str(facing_dir)+"\n")
			f.write("=================================\n")
'''
if __name__ == '__main__':
		configurations = get_configurations()
		print_configurations(configurations)

