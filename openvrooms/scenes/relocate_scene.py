import pybullet as p
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data

from gibson2.utils.utils import l2_distance
from transforms3d.euler import euler2quat

from openvrooms.objects.interactive_object import InteractiveObj
#from openvrooms.scenes.base_scene import Scene
from openvrooms.scenes.room_scene import RoomScene

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
#sys.path.insert(0, "../")
#from config import *
from openvrooms.config import *

from gibson2.utils.utils import quatToXYZW



class RelocateScene(RoomScene):
    """
    Room Environment relocate tasks
    """
    def __init__(self,
                 scene_id, 
                 n_interactive_objects=1
                 ):
        super(RelocateScene, self).__init__(scene_id=scene_id, fix_interactive_objects=False)

        
        #self.interative_object_obj_filename = '03337140_2f449bf1b7eade5772594f16694be05_object_alignedNew.obj'
        self.interative_object_obj_filename = '03337140_2f449bf1b7eade5772594f16694be05_object.obj'
        self.n_interactive_objects = n_interactive_objects
        

    def load(self):
        """
        Initialize scene
        """
        # load meta info
        self.load_scene_metainfo()

        # load layout
        self.load_layout()

        # load floor
        #self.load_floor()

        # load static objects
        self.load_static_objects()

        # load interactive objects
        self.load_interative_objects()
        
        # set layout and object's positions to their centroids
        self.reset_scene_object_positions_to_centroids()

        # randomly reset the poses of interative objects without collisions
        self.set_interative_object_initial_poses()

        # align the bottom of layout to z = 0 in world frame
        self.translate_scene([0, 0, -self.wall_bottom_z])

        # set initial z as goal z for all interactive objects
        self.set_goal_z()
        
        # print object poses
        #self.print_scene_info(interactive_only=True)

        # disable collision detection between fixed objects
        self.disable_collision_group()

        # print box dimension
        self.get_interactive_obj_dimension()

         # return static object ids, floor id, wall_id 
        return [self.floor_id] + self.static_object_ids

    
    def load_scene_metainfo(self):
        parser = SceneParser(scene_id=self.scene_id)
        pickle_path = os.path.join(metadata_path, str(self.scene_id)+'.pkl')
        parser.load_param(pickle_path)
        #parser.print_param()
        self.static_object_list, _, self.layout_list = parser.separate_static_interactive()
        
        
        interative_object_obj_filenames = [self.interative_object_obj_filename] * self.n_interactive_objects 
        for obj_filename in interative_object_obj_filenames:
            obj = SceneObj()
            obj.obj_path = obj_filename
            self.interative_object_list.append(obj)
            
        if self.layout_list is None:
            print('Error: No layout is found!') 
        else:
            print('Loaded meta info of layout')    
        
        print('Loaded meta info of %d static objects'%len(self.static_object_list))
        print('Loaded meta info of %d interactive objects'%len(self.interative_object_list))

            
    def get_interactive_obj_dimension(self):
        mesh = trimesh.load(os.path.join(self.scene_path, self.interative_object_obj_filename))
        bounds = mesh.bounds
        # bounds - axis aligned bounds of mesh
        # 2*3 matrix, min, max, x, y, z
        self.box_x_width = bounds[1][0] - bounds[0][0]
        self.box_y_width = bounds[1][1] - bounds[0][1]
        self.box_height =  bounds[1][2] - bounds[0][2]
        
        print("Box x width: %f"%(self.box_x_width))
        print("Box y width: %f"%(self.box_y_width))
        print("Box height: %f"%(self.box_height))
        print("-------------------------------------")
    
