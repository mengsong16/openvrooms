import pybullet as p
import os
import numpy as np
import trimesh
import copy

# base class of objects
class Object(object):
    def __init__(self):
        self.body_id = None
        self.loaded = False

    def load(self):
        # when call load for the second time
        if self.loaded:
            return self.body_id
        # when call load for the first time    
        self.body_id = self._load()
        self.loaded = True
        return self.body_id

    # object base pose in world frame
    def get_position(self):
        pos, _ = p.getBasePositionAndOrientation(self.body_id)
        return pos

    def get_orientation(self):
        """Return object orientation
        :return: quaternion in xyzw
        """
        _, orn = p.getBasePositionAndOrientation(self.body_id)
        return orn

    def get_position_orientation(self): 
        return p.getBasePositionAndOrientation(self.body_id)

    def set_position(self, pos):
        _, old_orn = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, pos, old_orn)

    def set_orientation(self, orn):
        old_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, old_pos, orn)

    def set_position_orientation(self, pos, orn):
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)

    # pos: [x,y]
    def set_xy_position_orientation(self, pos, orn):
        old_pos, _ = p.getBasePositionAndOrientation(self.body_id)
        p.resetBasePositionAndOrientation(self.body_id, [pos[0], pos[1], old_pos[2]], orn)    