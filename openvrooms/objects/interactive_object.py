import pybullet as p
import os
import numpy as np
import trimesh
import copy

from openvrooms.objects.base_object import Object

# only load urdf
class InteractiveObj(Object):
    """
    Interactive Objects are represented as a urdf, but doesn't have motors
    """
    def __init__(self, filename, fix_base=False):
        super(InteractiveObj, self).__init__()
        self.filename = filename
        self.body_id = None
        self.fix_base = fix_base
        self.volume = 0
        self.bbox = None
    
    # load from urdf and use material and colors from mtl
    def _load(self):
        # use the RGB color from the Wavefront OBJ file, instead of from the URDF file
        if self.fix_base:
            self.body_id = p.loadURDF(fileName=self.filename, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL, useFixedBase=1)
        else:    
            self.body_id = p.loadURDF(fileName=self.filename, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

        return self.body_id    

    def set_xy_position(self, x, y):
        old_pos, old_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, [x, y, old_pos[2]], old_orn)

    def get_xy_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return [pos[0], pos[1]]       

    def get_mesh(self):           
        obj_file = self.filename.replace(".urdf", ".obj")
        mesh = trimesh.load(obj_file)
        return mesh

    def get_mesh_com(self, center='geometric'):   
        mesh = self.get_mesh()

        if center == 'geometric':
            mesh_centroid = copy.deepcopy(mesh.centroid)
        else:
            mesh_centroid = mesh.center_mass

        return mesh_centroid
    
    def get_friction_coefficient(self):
        return p.getDynamicsInfo(self.body_id, -1)[1]   

    def get_mass(self):
        return p.getDynamicsInfo(self.body_id, -1)[0]   

    def set_friction_coefficient(self, mu):
        return p.changeDynamics(bodyUniqueId=self.body_id, linkIndex=-1, lateralFriction=mu)   

    def set_mass(self, mass):
        return p.changeDynamics(bodyUniqueId=self.body_id, linkIndex=-1, mass=mass)            

