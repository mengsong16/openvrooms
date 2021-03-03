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
#from openvrooms.scenes.room_scene import RoomScene
from openvrooms.scenes.relocate_scene import RelocateScene

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

from shutil import copyfile
import xml.etree.ElementTree as ET


class RelocateSceneDifferentObjects(RelocateScene):
    """
    Room Environment for relocate multiple objects with different materials
    """
    def __init__(self,
                 scene_id, 
                 n_interactive_objects=2, 
                 material_names=['Material__wood_hemlock', 'Material__steel_oxydized_bright']
                 ):
        super(RelocateSceneDifferentObjects, self).__init__(scene_id=scene_id, n_interactive_objects=n_interactive_objects)

        
        #self.interative_object_obj_filename = '03337140_2f449bf1b7eade5772594f16694be05_object_alignedNew.obj'
        self.base_obj_filename = '03337140_2f449bf1b7eade5772594f16694be05_object'
        self.n_interactive_objects = n_interactive_objects

        assert self.n_interactive_objects == len(material_names)

        self.interative_object_obj_filenames = []
        for i in list(range(self.n_interactive_objects)):
            target_urdf_filename = '%s_%d.urdf'%(self.base_obj_filename, i)
            target_obj_filename = '%s_%d.obj'%(self.base_obj_filename, i)
            target_mtl_filename = '%s_%d.mtl'%(self.base_obj_filename, i)
            target_diffuse_png_filename = '%s_%d_diffuse.png'%(self.base_obj_filename, i)
            target_rough_png_filename = '%s_%d_rough.png'%(self.base_obj_filename, i)
            self.interative_object_obj_filenames.append(target_obj_filename)

            #if not os.path.exists(os.path.join(self.scene_path, target_urdf_filename)):
            self.create_urdf_obj_mtl(target_urdf_filename, target_obj_filename, target_mtl_filename, target_diffuse_png_filename, target_rough_png_filename, material_names[i])

            print('Create new urdf file: %s'%(target_urdf_filename))
            print('Create new obj file: %s'%(target_obj_filename))
            print('Create new mtl file: %s'%(target_mtl_filename))
            print('------------------------------------------------')

    def create_urdf_obj_mtl(self, target_urdf_filename, target_obj_filename, target_mtl_filename, target_diffuse_png_filename, target_rough_png_filename, material_name): 
        source_obj_file = os.path.join(self.scene_path, self.base_obj_filename+'.obj')
        target_obj_file = os.path.join(self.scene_path, target_obj_filename)
        source_mtl_file = os.path.join(self.scene_path, self.base_obj_filename+'.mtl')
        target_mtl_file = os.path.join(self.scene_path, target_mtl_filename) 
        source_urdf_file = os.path.join(self.scene_path, self.base_obj_filename+'.urdf')
        target_urdf_file = os.path.join(self.scene_path, target_urdf_filename) 

        # copy target diffuse_png and rough_png files
        copyfile(os.path.join(brdf_dataset_path, material_name, 'tiled', 'diffuse_tiled.png'), os.path.join(self.scene_path, target_diffuse_png_filename))
        copyfile(os.path.join(brdf_dataset_path, material_name, 'tiled', 'rough_tiled.png'), os.path.join(self.scene_path, target_rough_png_filename))
        
        print('Create new diffuse png: %s'%(target_diffuse_png_filename))
        print('Create new rough file: %s'%(target_rough_png_filename))
        print('------------------------------------------------')

        # new mtl file with given material 
        with open(target_mtl_file, 'w') as wf:
            with open(source_mtl_file, 'r') as rf:
                lines = rf.readlines()
                for line in lines:
                    # diffuse
                    if "map_Kd" in line:
                        wf.write('map_Kd ' + target_diffuse_png_filename + '\n')
                    # rough    
                    elif "map_Pr" in line: 
                        wf.write('map_Pr ' + target_rough_png_filename + '\n') 
                    else:
                        wf.write(line)      
        
        # new obj file
        with open(target_obj_file, 'w') as wf:
            with open(source_obj_file, 'r') as rf:
                lines = rf.readlines()
                for line in lines:
                    # mtllib
                    if "mtllib" in line:
                        wf.write('mtllib ' + target_mtl_filename + '\n')
                    else:
                        wf.write(line)

        # new urdf file 
        urdf_tree = ET.parse(source_urdf_file)
        urdf_root = urdf_tree.getroot()
        visual_block = urdf_root.find('.//visual')
        visual_geometry_block = visual_block.find('geometry')
        mesh = visual_geometry_block.find('mesh')
        mesh.attrib['filename'] = target_obj_filename
        urdf_tree.write(target_urdf_file)              

    
    def load_scene_metainfo(self):
        parser = SceneParser(scene_id=self.scene_id)
        pickle_path = os.path.join(metadata_path, str(self.scene_id)+'.pkl')
        parser.load_param(pickle_path)
        #parser.print_param()
        self.static_object_list, _, self.layout_list = parser.separate_static_interactive()
        
        for obj_filename in self.interative_object_obj_filenames:
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
        for i, obj_filename in enumerate(self.interative_object_obj_filenames):
            #print(obj_filename)
            #print("******************************")
            mesh = trimesh.load(os.path.join(self.scene_path, obj_filename))
            bounds = mesh.bounds
            # bounds - axis aligned bounds of mesh
            # 2*3 matrix, min, max, x, y, z
            self.box_x_width = bounds[1][0] - bounds[0][0]
            self.box_y_width = bounds[1][1] - bounds[0][1]
            self.box_height =  bounds[1][2] - bounds[0][2]
            
            print("Object %d:"%(i))
            print("Box x width: %f"%(self.box_x_width))
            print("Box y width: %f"%(self.box_y_width))
            print("Box height: %f"%(self.box_height))
            print("-------------------------------------")
    
