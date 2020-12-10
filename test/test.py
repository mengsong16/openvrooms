import yaml
from openvrooms.robots.turtlebot import Turtlebot
from gibson2.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2.utils.utils import l2_distance
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.profiler import Profiler

import pytest
import pybullet as p
import numpy as np
import os
import gibson2

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

if __name__ == "__main__":
	aparser = argparse.ArgumentParser(description="Run run_demo.")
	aparser.add_argument("--id", default='scene0420_01', help="Scene ID")
	args = aparser.parse_args()
	
	test_scene(args.id, fix_interactive_objects=True)
	#test_layout()
	#test_robot()
	#test_object()

	#demo = DemoInteractive()
	#demo.run_demo()
