import yaml
from openvrooms.robots.turtlebot import Turtlebot
from gibson2.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2.utils.utils import l2_distance
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.profiler import Profiler
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.scene_base import Scene

import pytest
import pybullet as p
import numpy as np
import os
import gibson2

from gibson2.envs.igibson_env import iGibsonEnv


import sys
#sys.path.insert(0, "../")
from openvrooms.config import *

import time

import pybullet_data
import cv2

from scipy.spatial.transform import Rotation as R

import trimesh

import argparse

from openvrooms.scenes.room_scene import RoomScene
from openvrooms.scenes.relocate_scene import RelocateScene
from openvrooms.scenes.navigate_scene import NavigateScene
from openvrooms.objects.interactive_object import InteractiveObj

def test_object():
	p.connect(p.GUI)
	p.setGravity(0, 0, -9.8)
	p.setTimeStep(1./240.)

	floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
	p.loadMJCF(floor)
	
	scene_path = get_scene_path(scene_id='scene0420_01')
	chair1 = os.path.join(scene_path, '03001627_9231ef07326eae09b04cb542e2c50eb4_object_alignedNew.urdf')
	chair2 = os.path.join(scene_path, '03001627_fe57bad06e1f6dd9a9fe51c710ac111b_object_alignedNew.urdf')
	curtain = os.path.join(scene_path, 'curtain_4_object_alignedNew.urdf')

	obj1 = InteractiveObj(curtain)
	#obj1 = InteractiveObj(filename=curtain)
	obj1.load()
	#obj1.set_position([0,0,0.5])

	for _ in range(240000):  # at least 100 seconds
		p.stepSimulation()
		time.sleep(1./240.)

	p.disconnect()


def test_layout():
	p.connect(p.GUI)
	p.setGravity(0, 0, -9.8)
	p.setTimeStep(1./240.)

	#floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
	#p.loadMJCF(floor)

	scene = RoomScene(scene_id='scene0420_01', fix_interactive_objects=True)
	scene.load_scene_metainfo()
	scene.load_layout()


	for _ in range(240000):  # at least 100 seconds
		p.stepSimulation()
		time.sleep(1./240.)

	p.disconnect()

def test_relocate_scene(scene_id='scene0420_01', n_interactive_objects=1):
	time_step = 1./240. 
	p.connect(p.GUI)
	p.setGravity(0, 0, -9.8)
	p.setTimeStep(time_step)
	
	scene = RelocateScene(scene_id=scene_id, n_interactive_objects=n_interactive_objects)
	scene.load()
	
	robot_config = parse_config(os.path.join(config_path, "turtlebot_relocate.yaml"))
	turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file) 

	turtlebot.load()
	
	turtlebot.set_position([0, 0, 0])
	turtlebot.robot_specific_reset()
	turtlebot.keep_still()
	
	for _ in range(2400000):  # at least 100 seconds
		 p.stepSimulation()
		 time.sleep(1./240.)

	p.disconnect()

def test_navigate_scene(scene_id='scene0420_01', n_obstacles=1):
	time_step = 1./240. 
	p.connect(p.GUI)
	p.setGravity(0, 0, -9.8)
	p.setTimeStep(time_step)
	
	scene = NavigateScene(scene_id=scene_id, n_obstacles=n_obstacles)
	scene.load()
	
	robot_config = parse_config(os.path.join(config_path, "turtlebot_navigate.yaml"))
	turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file) 

	turtlebot.load()
	
	turtlebot.set_position([0, 0, 0])
	turtlebot.robot_specific_reset()
	turtlebot.keep_still()
	
	for _ in range(2400000):  # at least 100 seconds
		 p.stepSimulation()
		 time.sleep(1./240.)

	p.disconnect()

def test_scene(scene_id='scene0420_01', fix_interactive_objects=True):
	time_step = 1./240. 
	p.connect(p.GUI)
	p.setGravity(0, 0, -9.8)
	p.setTimeStep(time_step)

	#floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
	#p.loadMJCF(floor)
	scene = RoomScene(scene_id=scene_id, load_from_xml=True, fix_interactive_objects=fix_interactive_objects, empty_room=False)
	scene.load()
	
	#scene.change_interactive_objects_dynamics(mass=100)
	
	robot_config = parse_config(os.path.join(config_path, "turtlebot_interactive_demo.yaml"))
	turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file) 

	turtlebot.load()
	
	turtlebot.set_position([0, 0, 0])
	turtlebot.robot_specific_reset()
	turtlebot.keep_still()
	
	for _ in range(2400000):  # at least 100 seconds
		 p.stepSimulation()
		 time.sleep(1./240.)

	p.disconnect()

def test_robot():
	p.connect(p.GUI)
	p.setGravity(0,0,-9.8)
	p.setTimeStep(1./240.)

	floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
	p.loadMJCF(floor)

	robot_config = parse_config(os.path.join(config_path, "turtlebot_interactive_demo.yaml"))
	turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file) 

	turtlebot.load()
	turtlebot.set_position([0, 0, 0])
	turtlebot.robot_specific_reset()
	turtlebot.keep_still() 

	print(turtlebot.get_position())
	print(turtlebot.get_orientation())

	
	for _ in range(24000):  # move with small random actions for 10 seconds
		action = np.random.uniform(-1, 1, turtlebot.action_dim)
		turtlebot.apply_action(action)
		p.stepSimulation()
		time.sleep(1./240.0)
	
	p.disconnect()

class DemoInteractive(object):
	def __init__(self):
		return

	def run_demo(self):
		s = Simulator(mode='pbgui', image_width=700, image_height=700)

		robot_config = parse_config(os.path.join(config_path, "turtlebot_interactive_demo.yaml"))
		turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file)

		scene = RoomScene(scene_id='scene0420_01', fix_interactive_objects=True)

		s.import_scene(scene)
		s.import_robot(turtlebot)
		
		time_step = 1./350.
		
		for i in range(10000):
			turtlebot.apply_action([0.1,0.5])
			s.step()
		
		s.disconnect()

def test_igibson_floor():
	gibson_datapath = "/Users/meng/Documents/iGibson/gibson2/data/ig_dataset"
	floor_path = "scenes/Ihlen_0_int/shape/collision/floor_cm.obj"
	floor_mesh = trimesh.load(os.path.join(gibson_datapath, floor_path))
	bounds = floor_mesh.bounds
	# bounds - axis aligned bounds of mesh
	# 2*3 matrix, min, max, x, y, z
	# assume z up
	ground_z = bounds[1][2]
	bottom_z = bounds[0][2]
	x_range = [bounds[0][0], bounds[1][0]]
	y_range = [bounds[0][1], bounds[1][1]]

	print("Layout range: x=%s, y=%s"%(x_range, y_range))
	print("Ground z: %f"%(ground_z))
	print("Bottom z: %f"%(bottom_z))

def floor_collision_detection(robot_id, floor_id):
	collision_links = list(p.getContactPoints(bodyA=robot_id, bodyB=floor_id))
	
	return len(collision_links) > 0 

def test_igibson_floor_robot_collision():
	config_filename = os.path.join(
		gibson2.root_path, 'test', 'test_house.yaml')
	env = iGibsonEnv(config_file=config_filename, mode='headless')

	floor_id = env.simulator.scene.floor_body_ids[0]
	robot_id = env.robots[0].robot_ids[0]

	print(floor_id)
	print(robot_id)

	print(str(env.collision_ignore_body_b_ids))

	floor_collision_steps = 0
	for j in range(2):
		env.reset()
		print('After reset')
		print(str(env.collision_ignore_body_b_ids))
		for i in range(300):
			action = env.action_space.sample()
			env.step(action)
			floor_collision_steps += floor_collision_detection(robot_id, floor_id)
			print('----------------------------------')
			print(env.collision_step)
			print(floor_collision_steps)
	
	env.close()

if __name__ == "__main__":
	aparser = argparse.ArgumentParser(description="Run run_demo.")
	aparser.add_argument("--id", default='scene0420_01', help="Scene ID")
	args = aparser.parse_args()
	
	#test_relocate_scene(args.id, n_interactive_objects=1)
	test_navigate_scene(args.id, n_obstacles=1)
	#test_scene(args.id, fix_interactive_objects=False)
	#test_layout()
	#test_robot()
	#test_object()
	'''
	demo = DemoInteractive()
	demo.run_demo()
	'''
	#test_igibson_floor()
	#test_igibson_floor_robot_collision()