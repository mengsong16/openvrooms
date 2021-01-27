
from openvrooms.data_processor.xml_parser import SceneParser, SceneObj
import pybullet as p
import os
import pybullet_data
from transforms3d.euler import euler2quat
from openvrooms.objects.interactive_object import InteractiveObj

from openvrooms.scenes.room_scene import RoomScene
from openvrooms.scenes.relocate_scene import RelocateScene
from openvrooms.scenes.navigate_scene import NavigateScene

import numpy as np
from PIL import Image
import cv2
import networkx as nx
import pickle
import logging
from openvrooms.data_processor.xml_parser import SceneParser, SceneObj
import trimesh
import copy
import random
import math
import itertools
import sys
import shutil
import time
from openvrooms.data_processor.adapted_object_urdf import ObjectUrdfBuilder
from openvrooms.config import *
#from gibson2.utils.utils import quatToXYZW
from gibson2.utils.utils import *
from openvrooms.robots.turtlebot import Turtlebot

class ComponentBBox(object):
	# 2*3 matrix, min, max, x, y, z
	def __init__(self, x_min, x_max, y_min, y_max, z_min, z_max):
		self.x = np.array([x_min, x_max])
		self.y = np.array([y_min, y_max])
		self.z = np.array([z_min, z_max])

	# generate x,y coordinates of the 2D bounding box in world frame (not floor frame)
	def get_coords(self):
			'''
			Return the vertices of the bounding box, in order of BL,BR,TR,TL
			x: left, right
			y: bottom, top
			'''
			'''
			x = self.edge_x / 2.
			y = self.edge_y / 2.
			return np.array([self.center - x - y,
							self.center + x - y,
							self.center + x + y,
							self.center - x + y])
			'''
			return np.array([[self.x[0], self.y[0]], 
							[self.x[1], self.y[0]],
							[self.x[1], self.y[1]],
							[self.x[0], self.y[1]]])

	def gen_cube_obj(self, file_path, is_color=False, should_save=True):
		vertices = []
		a,b,c,d = self.get_coords()
		for x,y in [a,b,d,c]:
			vertices.append((x,y,self.z[1]))
		for x,y in [a,b,d,c]:
			vertices.append((x,y,self.z[0]))
		c=np.random.rand(3)
		faces = [(1,2,3),
				 (2,4,3),
				 (1,3,5),
				 (3,7,5),
				 (1,5,2),
				 (5,6,2),
				 (2,6,4),
				 (6,8,4),
				 (4,8,7),
				 (7,3,4),
				 (6,7,8),
				 (6,5,7),
				]
		faces = [(*f, -1) for f in faces]

		if should_save:
			with open(file_path, 'w') as fp:
				for v in vertices:
					if is_color:
						v1,v2,v3 = v
						fp.write('v {} {} {} {} {} {}\n'.format(v1, v2, v3, *c))
					else:
						v1,v2,v3 = v
						fp.write('v {} {} {}\n'.format(v1, v2, v3))

				for f in faces:
					fp.write('f {} {} {}\n'.format(*f[:-1]))

		return vertices, faces
		

def load_original_layout(scene_id, scene_path):
	# load layout metadata
	parser = SceneParser(scene_id)
	pickle_path = os.path.join(metadata_path, str(scene_id)+'.pkl')
	parser.load_param(pickle_path)
		
	_, _, layout_meta = parser.separate_static_interactive()

	
	if layout_meta is None:
		print('Error: No layout is found!') 
	else:
		print('Loaded meta info of layout')  

	obj_file_name = layout_meta.obj_path
	urdf_file_name = os.path.splitext(obj_file_name)[0] + '.urdf'
	layout_urdf_file = os.path.join(scene_path, urdf_file_name)
	
	# load layout urdf
	#layout_id = p.loadURDF(fileName=layout_urdf_file, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL, useFixedBase=1)
	#print('Layout urdf loaded: %s'%(layout_urdf_file))
	
	# load mesh
	layout_mesh = trimesh.load(os.path.join(scene_path, obj_file_name))

	return layout_mesh

def generate_bbox_floor_obj(layout_mesh, scene_path):	
	# bounds - axis aligned bounds of mesh
	# 2*3 matrix, min, max, x, y, z
	layout_bounds = layout_mesh.bounds

	floor_thickness = 0.2
	#print(layout_bounds)
	
	floor_bbox = ComponentBBox(layout_bounds[0][0], layout_bounds[1][0], layout_bounds[0][1], layout_bounds[1][1], -floor_thickness, 0.0) 
	#print(floor_bbox.x)
	#print(floor_bbox.y)
	#print(floor_bbox.z)

	# generate floor .obj file
	floor_obj_path = os.path.join(scene_path, "floor.obj")
	floor_bbox.gen_cube_obj(file_path=floor_obj_path, is_color=False, should_save=True)
	print('Generated floor obj file')

	# copy .obj to vhach.obj
	floor_vhacd_obj_path = os.path.join(scene_path, "floor_vhacd.obj")
	shutil.copyfile(floor_obj_path, floor_vhacd_obj_path)	
	print('Generated floor vhacd obj file')
	'''
	'''

def generate_floor_urdf(scene_path):	
	urdf_prototype_file = os.path.join(metadata_path, 'urdf_prototype.urdf') # urdf template
	log_file = os.path.join(scene_path, "vhacd_log.txt")

	builder = ObjectUrdfBuilder(scene_path, log_file=log_file, urdf_prototype=urdf_prototype_file)
	floor_obj_path = os.path.join(scene_path, "floor.obj")
	builder.build_urdf(filename=floor_obj_path, force_overwrite=True, decompose_concave=True, force_decompose=False, mass=100, center=None)	 #'geometric'
	print('Generated floor urdf file')

def floor_collision_detection(robot_id, floor_id):
	collision_links = list(p.getContactPoints(bodyA=robot_id, bodyB=floor_id))
	for item in collision_links:
		print('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(item[1], item[2], item[3], item[4]))
	
	return len(collision_links) > 0 

def load_floor(scene_path):
	floor_urdf_file = os.path.join(scene_path, "floor.urdf")
	floor_id = p.loadURDF(fileName=floor_urdf_file, useFixedBase=1)

	#p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
	#floor_id = p.loadURDF("plane.urdf")

	# change floor color
	p.changeVisualShape(objectUniqueId=floor_id, linkIndex=-1, rgbaColor=[0.86,0.86,0.86,1])

	floor_pos, floor_orn = p.getBasePositionAndOrientation(floor_id)

	print("Floor position: %s"%str(floor_pos))
	print("Floor orientation: %s"%str(floor_orn))

	return floor_id

def test_floor_urdf(scene_path):
	time_step = 1./240. 
	p.connect(p.GUI) # load with pybullet GUI
	p.setGravity(0, 0, -9.8)
	p.setTimeStep(time_step)

	# load floor or scene
	floor_id = load_floor(scene_path)
	'''
	scene = NavigateScene(scene_id='scene0420_01', n_obstacles=0)
	scene.load()
	floor_id = scene.floor_id
	'''

	# load robot
	robot_config = parse_config(os.path.join(config_path, "turtlebot_interactive_demo.yaml"))
	turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file) 

	robot_ids = turtlebot.load()
	robot_id = robot_ids[0]

	turtlebot.set_position([0, 0, 0])
	turtlebot.robot_specific_reset()
	turtlebot.keep_still() 

	collision_counter = 0
	# start simulation
	
	# keep still
	for _ in range(100):
		p.stepSimulation()
		#collision_counter += floor_collision_detection(robot_id, floor_id)
		time.sleep(time_step) # this is just for visualization, could be removed without affecting avoiding initial collisions
	
	
	# move    
	time_step_n = 0
	for _ in range(50):  # at least 100 seconds
		action = np.random.uniform(-1, 1, turtlebot.action_dim)
		turtlebot.apply_action(action)
		p.stepSimulation()
		time_step_n += 1
		print('----------------------------------------')
		print('time step: %d'%(time_step_n))
		collision_counter += floor_collision_detection(robot_id, floor_id)
		time.sleep(time_step)

	print("Collision steps:%d"%(collision_counter))
	
	p.disconnect()

if __name__ == '__main__':
	scene_id='scene0420_01'
	scene_path = get_scene_path(scene_id)

	'''
	layout_mesh = load_original_layout(scene_id, scene_path)
	generate_bbox_floor_obj(layout_mesh, scene_path)
	generate_floor_urdf(scene_path)
	'''
	test_floor_urdf(scene_path)
	