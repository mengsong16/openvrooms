import pybullet as p
import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
import pybullet_data

from gibson2.utils.utils import l2_distance

from openvrooms.objects.interactive_object import InteractiveObj
#from openvrooms.scenes.base_scene import Scene
from gibson2.scenes.scene_base import Scene

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
import itertools

import sys
#sys.path.insert(0, "../")
#from config import *
from openvrooms.config import *
from gibson2.utils.utils import quatToXYZW
from transforms3d.euler import euler2quat


class RoomScene(Scene):
    """
    Openroom Environment room scenes
    """
    def __init__(self,
                 scene_id, 
                 load_from_xml=True,
                 fix_interactive_objects=False,
                 empty_room=False,
                 multi_band=False
                 ):
        
        logging.info("Room scene: {}".format(scene_id))
        self.scene_id = scene_id
        self.scene_path = get_scene_path(self.scene_id)
        self.is_interactive = True 

        # meta info
        self.static_object_list = []  # metainfo list of static objects except layout
        self.interative_object_list = [] # metainfo list of interactive objects
        self.layout_list = None # metainfo of layout

        self.multi_band = multi_band

        # obj file names
        if self.multi_band:
            self.floor_obj_file_name = []
        else:    
            self.floor_obj_file_name = None

        self.wall_obj_file_name = None
        self.ceiling_obj_file_name = None

        self.interative_objects = [] # a list of InteractiveObj
        self.static_object_ids = [] # a list of urdf ids of static objects, including ceilings, window, door, ...

        if self.multi_band:
            self.floor_id = []
        else:    
            self.floor_id = None # urdf id of floor

        self.wall_id = None # urdf id of walls
        self.ceiling_id = None # urdf id of ceiling

        # original z coordinate of wall bottom
        self.wall_bottom_z = -1.296275
        self.room_height = None
        self.x_range = [-3.4184, 3.5329]
        self.y_range = [-2.8032, 2.7222]

        self.load_from_xml = load_from_xml

        self.fix_interactive_objects = fix_interactive_objects

        self.empty_room = empty_room


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
        
        #self.print_scene_info(interactive_only=True)

        # reset object centers
        self.reset_scene_object_positions_to_centroids()

        # align the bottom of layout to z = 0 in world frame
        self.translate_scene([0, 0, -self.wall_bottom_z])
        
        # print object poses
        #self.print_scene_info(interactive_only=True)

        # disable collision detection between fixed objects
        self.disable_collision_group()

         # return static object ids, floor id, wall_id 
        if self.multi_band:
            return self.floor_id + self.static_object_ids
        else:    
            return [self.floor_id] + self.static_object_ids

    # load interactive objects
    def load_interative_objects(self):
         
        for obj_meta in self.interative_object_list:
            obj_file_name = obj_meta.obj_path
            urdf_file_name = os.path.splitext(obj_file_name)[0] + '.urdf'
            urdf_file = os.path.join(self.scene_path, urdf_file_name)
            urdf_path = os.path.join(self.scene_path, urdf_file)
            if os.path.exists(urdf_path):
                obj = InteractiveObj(filename=urdf_path, fix_base=self.fix_interactive_objects)
                # load from the object urdf file
                obj.load()
                self.interative_objects.append(obj)
            else:
                print('Error: File Not Exists: %s'%(urdf_path))    

        print('Object urdf loaded: %d interactive objects'%(len(self.interative_objects)))

             
    # load static objects
    def load_static_objects(self):
        for obj_meta in self.static_object_list:
            obj_file_name = obj_meta.obj_path
            urdf_file_name = os.path.splitext(obj_file_name)[0] + '.urdf'
            urdf_file = os.path.join(self.scene_path, urdf_file_name)
            urdf_path = os.path.join(self.scene_path, urdf_file)

            #print(urdf_file)
            if os.path.exists(urdf_path):
                # load from the object urdf file
                urdf_id = p.loadURDF(fileName=urdf_path, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL, useFixedBase=1)
                self.static_object_ids.append(urdf_id)
                #print(urdf_id)
            else:
                print('Error: File Not Exists: %s'%(urdf_path))    

        print('Object urdf loaded: %d static objects'%(len(self.static_object_ids)))

    # load layout --> load walls
    # compute room x,y range and height
    # get wall bottom z to translate the scene
    def split_name_floor_wall_ceiling(self):
        for obj_meta in self.layout_list:
            name = obj_meta.obj_path
            if 'floor' in name:
                if self.multi_band:
                    self.floor_obj_file_name.append(name)
                else:    
                    self.floor_obj_file_name = name
            elif 'wall' in name: 
                self.wall_obj_file_name = name
            elif 'ceiling' in name: 
                self.ceiling_obj_file_name = name 

        #return floor_obj_file_name, wall_obj_file_name, ceiling_obj_file_name              

    def get_split_urdf(self, obj_file_name):
        urdf_file_name = os.path.splitext(obj_file_name)[0] + '.urdf'
        layout_urdf_file = os.path.join(self.scene_path, urdf_file_name)
        
        return p.loadURDF(fileName=layout_urdf_file, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL, useFixedBase=1)

    def load_layout(self):
        self.split_name_floor_wall_ceiling()
        
        #obj_file_name = self.layout_list.obj_path
        
        #floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        #self.layout_id = p.loadMJCF(floor)
        if self.multi_band:
            for floor_name in self.floor_obj_file_name:
                self.floor_id.append(self.get_split_urdf(floor_name))
        else:    
            print(self.floor_obj_file_name)
            self.floor_id = self.get_split_urdf(self.floor_obj_file_name)

        self.wall_id = self.get_split_urdf(self.wall_obj_file_name)
        self.ceiling_id = self.get_split_urdf(self.ceiling_obj_file_name)

        #print('Layout urdf loaded: %s'%(layout_urdf_file))
        #print('Layout id: %d'%(self.layout_id))
        print('Layout urdf loaded')
        if self.multi_band:
            print('Layout ids: floor: %s, wall: %d, ceiling: %d'%(str(self.floor_id), self.wall_id, self.ceiling_id))
        else:    
            print('Layout ids: floor: %d, wall: %d, ceiling: %d'%(self.floor_id, self.wall_id, self.ceiling_id))

        # compute z coordinate, x and y range of the ground
        #filename, _ = os.path.splitext(obj_file_name)
        #obj_file_name = filename + "_vhacd.obj"
        #print(obj_file_name)
        mesh = trimesh.load(os.path.join(self.scene_path, self.wall_obj_file_name))
        bounds = mesh.bounds
        # bounds - axis aligned bounds of mesh
        # 2*3 matrix, min, max, x, y, z
        self.wall_bottom_z = bounds[0][2]
        self.room_height =  bounds[1][2] - bounds[0][2]
        self.x_range = [bounds[0][0], bounds[1][0]]
        self.y_range = [bounds[0][1], bounds[1][1]]

        print("Layout range: x=%s, y=%s"%(self.x_range, self.y_range))
        print("Room height: %f"%(self.room_height))
        
    # load generated floor
    # upper surface of floor has z=0
    # discarded !!!
    def load_floor(self):
        floor_urdf_file = os.path.join(self.scene_path, "floor.urdf")
        self.floor_id = p.loadURDF(fileName=floor_urdf_file, useFixedBase=1)
        # change floor color
        p.changeVisualShape(objectUniqueId=self.floor_id, linkIndex=-1, rgbaColor=[0.86,0.86,0.86,1])

        print('Floor urdf loaded: %s'%(floor_urdf_file))
        print('Floor id: %d'%(self.floor_id))


    def load_scene_metainfo(self):
        parser = SceneParser(scene_id=self.scene_id)
        pickle_path = os.path.join(metadata_path, str(self.scene_id)+'.pkl')
        parser.load_param(pickle_path)
        #parser.print_param()
        self.static_object_list, self.interative_object_list, self.layout_list = parser.separate_static_interactive()
        #print(self.layout_list)

        if not self.load_from_xml:
            self.interative_object_list = []
            # predefined a subset of interactive objects
            if not self.empty_room:
                #interative_object_obj_filenames = ['03001627_fe57bad06e1f6dd9a9fe51c710ac111b_object_alignedNew.obj']  # brown chair
                #interative_object_obj_filenames = ['03337140_2f449bf1b7eade5772594f16694be05_object_alignedNew.obj'] # cabinet
                interative_object_obj_filenames = ['03337140_2f449bf1b7eade5772594f16694be05_object.obj']
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

    # given object's urdf id, translate it
    def translate_object(self, urdf_id, translate):
        old_translate, old_orn = p.getBasePositionAndOrientation(urdf_id)
        new_translate = np.asarray(old_translate) + np.asarray(translate)
        # the result is a float list
        p.resetBasePositionAndOrientation(urdf_id, new_translate.tolist(), old_orn)

    def translate_scene(self, translate):
        print("Translate the entire scene by %s"%(translate))

        # translate wall
        if self.wall_id is not None:
            self.translate_object(self.wall_id, translate)

        # translate ceiling
        if self.ceiling_id is not None:
            self.translate_object(self.ceiling_id, translate)    

        # translate static objects
        for urdf_id in self.static_object_ids:
            self.translate_object(urdf_id, translate)

        # translate interactive objects
        for obj in self.interative_objects:
            self.translate_object(obj.body_id, translate)

    def change_interactive_objects_dynamics(self, mass):
        # transform interactive objects
        for obj in self.interative_objects:
            p.changeDynamics(bodyUniqueId=obj.body_id, linkIndex=-1, mass=mass)

    # print info of each object in the scene (com, pose, ...)
    def print_scene_info(self, interactive_only=False):
        if not interactive_only:
            if self.floor_id is not None:
                if self.multi_band:
                    for i in list(range(len(self.floor_id))):
                        self.get_metric_centroid_physics_pose(obj_file_name=self.floor_obj_file_name[i], urdf_id=self.floor_id[i])
                else:    
                    self.get_metric_centroid_physics_pose(obj_file_name=self.floor_obj_file_name, urdf_id=self.floor_id)

            if self.wall_id is not None:
                self.get_metric_centroid_physics_pose(obj_file_name=self.wall_obj_file_name, urdf_id=self.wall_id) 

            if self.ceiling_id is not None:
                self.get_metric_centroid_physics_pose(obj_file_name=self.ceiling_obj_file_name, urdf_id=self.ceiling_id)       

            n_static_objs = len(self.static_object_ids)
            if n_static_objs > 0:
                for i in list(range(n_static_objs)):
                    self.get_metric_centroid_physics_pose(obj_file_name=self.static_object_list[i].obj_path, urdf_id=self.static_object_ids[i])

        n_interactive_objs = len(self.interative_objects) 
        if n_interactive_objs > 0:
            for i in list(range(n_interactive_objs)):
                self.get_metric_centroid_physics_pose(obj_file_name=self.interative_object_list[i].obj_path, urdf_id=self.interative_objects[i].body_id)   

    # no need to move floors
    def reset_layout_static_object_positions_to_centroids(self):
        #if self.floor_id is not None:
        #    self.set_mesh_centroid_as_position(obj_file_name="floor.obj", urdf_id=self.floor_id)

        if self.wall_id is not None:
            self.set_mesh_centroid_as_position(obj_file_name=self.wall_obj_file_name, urdf_id=self.wall_id)   

        if self.ceiling_id is not None:
            self.set_mesh_centroid_as_position(obj_file_name=self.ceiling_obj_file_name, urdf_id=self.ceiling_id)         

        n_static_objs = len(self.static_object_ids)
        if n_static_objs > 0:
            for i in list(range(n_static_objs)):
                self.set_mesh_centroid_as_position(obj_file_name=self.static_object_list[i].obj_path, urdf_id=self.static_object_ids[i])

    def reset_scene_object_positions_to_centroids(self):
        self.reset_layout_static_object_positions_to_centroids()

        # reset interactive objects to com
        n_interactive_objs = len(self.interative_objects) 
        if n_interactive_objs > 0:
            for i in list(range(n_interactive_objs)):
                self.set_mesh_centroid_as_position(obj_file_name=self.interative_object_list[i].obj_path, urdf_id=self.interative_objects[i].body_id)   

    def get_floor_friction_coefficient(self):
        if self.multi_band:
            fc = []
            for fid in self.floor_id:
                fc.append(p.getDynamicsInfo(fid, -1)[1])
            return fc    
        else:
            return p.getDynamicsInfo(self.floor_id, -1)[1] 

    def set_floor_friction_coefficient(self, mu):
        if self.multi_band:
            assert mu.shape[0] == len(self.floor_id)
            for i in list(range(len(self.floor_id))):
                p.changeDynamics(bodyUniqueId=self.floor_id[i], linkIndex=-1, lateralFriction=mu[i])
        else:    
            return p.changeDynamics(bodyUniqueId=self.floor_id, linkIndex=-1, lateralFriction=mu)     

    def get_metric_centroid_physics_pose(self, obj_file_name, urdf_id, center='geometric'):
        mesh = trimesh.load(os.path.join(self.scene_path, obj_file_name))

        if center == 'geometric':
            mesh_centroid = copy.deepcopy(mesh.centroid)
        else:
            mesh_centroid = mesh.center_mass

        obj_pose = p.getBasePositionAndOrientation(urdf_id)  

        dynamics_info = p.getDynamicsInfo(urdf_id, -1)
        
        mass = dynamics_info[0]
        lateral_friction = dynamics_info[1]
        local_inertia_diagonal = dynamics_info[2]

        inertia_position = dynamics_info[3]
        inertia_orientation = dynamics_info[4]


        visual_info = p.getVisualShapeData(objectUniqueId=urdf_id)
        visual_position = visual_info[0][5]
        visual_orientation = visual_info[0][6]
        
        collision_info = p.getCollisionShapeData(objectUniqueId=urdf_id, linkIndex=-1)
        collision_position = collision_info[0][5]
        collision_orientation = collision_info[0][6]

        # can only be called after simulation starts
        #link_state = p.getLinkState(objectUniqueId=urdf_id, linkIndex=-1)
        #linkWorldPosition = link_state[0]
        #worldLinkFramePosition = link_state[4]

        print("-------------------------------------------------------------------")
        print("Obj file: %s"%(obj_file_name))
        print("URDF id: %s"%(str(urdf_id)))
        print("Mesh centroid from .obj frame: %s"%(str(mesh_centroid)))
        print("Object pose in world frame: %s, %s"%(str(obj_pose[0]), str(obj_pose[1])))
        print("Bullet visual pose: %s, %s"%(str(visual_position), str(visual_orientation)))
        print("Bullet collision pose: %s, %s"%(str(collision_position), str(collision_orientation)))
        print("Bullet intertia pose: %s, %s"%(str(inertia_position), str(inertia_orientation)))
        print("Mass (kg): %f"%(mass))
        print("Friction coefficient: %f"%(lateral_friction))
        print("Local inertia diagonal: %s"%(str(local_inertia_diagonal)))
        #print("Position of center of mass: %s"%(str(linkWorldPosition)))
        #print("World position of the URDF link frame: %s"%(str(worldLinkFramePosition)))
        print("-------------------------------------------------------------------")


    def set_mesh_centroid_as_position(self, obj_file_name, urdf_id, center='geometric'): 
        mesh = trimesh.load(os.path.join(self.scene_path, obj_file_name))
        if center == 'geometric':
            mesh_centroid = copy.deepcopy(mesh.centroid)
        else:
            mesh_centroid = mesh.center_mass
        

        _, old_orn = p.getBasePositionAndOrientation(urdf_id)
        p.resetBasePositionAndOrientation(bodyUniqueId=urdf_id, posObj=mesh_centroid, ornObj=old_orn)  


    def get_interative_object_pb_ids(self):
        # collect urdf ids of interactive objects
        ids = []
        for obj in self.interative_objects:
            ids.append(obj.body_id) 

        return ids  

    # set initial z as goal z for all interactive objects
    def set_goal_z(self):
        for obj in self.interative_objects:
            obj_pos = obj.get_position()
            obj.goal_z = obj_pos[2]
            
     # ensure no collision, set x,y position
    def set_interative_object_initial_poses(self):
        # divide x,y space into grid
        self.x_cell_n = 4
        self.y_cell_n = 2

        self.x_cell_size = float(self.x_range[1] - self.x_range[0]) / float(self.x_cell_n)
        self.y_cell_size = float(self.y_range[1] - self.y_range[0]) / float(self.y_cell_n)

        #print(self.x_cell_size)
        #print(self.y_cell_size)

        full_cell_list = list(itertools.product(np.arange(self.x_cell_n).tolist(), np.arange(self.y_cell_n).tolist()))

        assert self.n_interactive_objects <= self.x_cell_n * self.y_cell_n

        # random sampling without replacement
        sel_cell_list = random.sample(full_cell_list, self.n_interactive_objects)

        #print(sel_cell_list)

        s = ""

        for i, obj in enumerate(self.interative_objects):
            x_coord = self.x_range[0] + (sel_cell_list[i][0] + 0.5) * self.x_cell_size
            y_coord = self.y_range[0] + (sel_cell_list[i][1] + 0.5) * self.y_cell_size

            #print(x_coord)
            #print(y_coord)

            obj.set_xy_position(x_coord, y_coord)

            s += "[%.1f,%.1f] "%(x_coord, y_coord)

        
        print("Object positions when initially loading the scene: "+s)

    # reset the poses of interactive objects according to given values
    # input pos: [x,y]
    # input orn: eular angles
    def reset_interactive_object_poses(self, obj_pos_list, obj_orn_list): 
        ''' 
        n_interactive_objs = len(self.interative_objects) 

        for i in list(range(n_interactive_objs)):
            #print(obj_orn_list[i])
            p.resetBasePositionAndOrientation(bodyUniqueId=self.interative_objects[i].body_id, posObj=obj_pos_list[i], ornObj=quatToXYZW(euler2quat(obj_orn_list[i][0], obj_orn_list[i][1], obj_orn_list[i][2])))
        '''
        for i, obj in enumerate(self.interative_objects):    
            obj.set_position_orientation([obj_pos_list[i][0], obj_pos_list[i][1], obj.goal_z], quatToXYZW(euler2quat(obj_orn_list[i][0], obj_orn_list[i][1], obj_orn_list[i][2]), 'wxyz'))
            #obj.set_xy_position_orientation(obj_pos_list[i], quatToXYZW(euler2quat(obj_orn_list[i][0], obj_orn_list[i][1], obj_orn_list[i][2]), 'wxyz'))
            #print(obj_pos_list[i])   

    # disable collision detection between walls, floor, static objects and fixed interactive objects
    def disable_collision_group(self):
        fixed_body_ids = self.static_object_ids

        if self.floor_id is not None:
            if self.multi_band:
                for fid in self.floor_id:
                    fixed_body_ids.append(fid)
            else:    
                fixed_body_ids.append(self.floor_id)

        if self.wall_id is not None:
            fixed_body_ids.append(self.wall_id)    

        if self.fix_interactive_objects:
            fixed_body_ids += self.get_interative_object_pb_ids()

        # disable collision between the fixed links of the fixed objects
        for i in range(len(fixed_body_ids)):
            for j in range(i + 1, len(fixed_body_ids)):
                # link_id = 0 is the base link that is connected to the world
                # by a fixed link
                p.setCollisionFilterPair(
                    fixed_body_ids[i],
                    fixed_body_ids[j],
                    -1, -1, enableCollision=0)    

        print("Disabled collision detection between %d objects"%(len(fixed_body_ids)))

