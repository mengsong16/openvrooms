from gibson2.sensors.sensor_base import BaseSensor
from gibson2.sensors.vision_sensor import VisionSensor
from gibson2.sensors.dropout_sensor_noise import DropoutSensorNoise

import numpy as np
import os
import gibson2
from collections import OrderedDict

from gibson2.utils.mesh_util import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, safemat2quat, xyzw2wxyz, ortho, transform_vertex


class ExternalVisionSensor(VisionSensor):
    """
    Third person view Vision sensor (including rgb, rgb_filled, depth, 3d, seg, normal, optical flow, scene flow)
    not from robot's first person view point
    """

    def __init__(self, env, modalities, camera_pos=[0, 0, 1.2], camera_view_direction=[1, 0, 0]):
        super(ExternalVisionSensor, self).__init__(env, modalities)
        self.camera_pos = np.array(camera_pos)
        self.camera_view_direction = np.array(camera_view_direction)

    # camera_orn: quarternion x,y,z,w
    # camera_pos: x,y,z
    def render_third_person_view_cameras(self, env, modes=('rgb')):
        """
        Render robot camera images

        :return: a list of frames (number of modalities x number of robots)
        """
        frames = []

        env.simulator.renderer.set_camera(self.camera_pos, self.camera_pos + self.camera_view_direction, [0, 0, 1], cache=True)
        hidden_instances = []
        
        for item in env.simulator.renderer.render(modes=modes):
            frames.append(item)

        return frames

    def get_obs(self, env):
        """
        Get vision sensor reading

        :return: vision sensor reading
        """

        #raw_vision_obs = env.simulator.renderer.render_robot_cameras(modes=self.raw_modalities)
        raw_vision_obs = self.render_third_person_view_cameras(env, modes=self.raw_modalities)

        raw_vision_obs = {
            mode: value
            for mode, value in zip(self.raw_modalities, raw_vision_obs)
        }

        vision_obs = OrderedDict()
        if 'rgb' in self.modalities:
            vision_obs['rgb'] = self.get_rgb(raw_vision_obs)
        if 'rgb_filled' in self.modalities:
            vision_obs['rgb_filled'] = self.get_rgb_filled(raw_vision_obs)
        if 'depth' in self.modalities:
            vision_obs['depth'] = self.get_depth(raw_vision_obs)
        if 'pc' in self.modalities:
            vision_obs['pc'] = self.get_pc(raw_vision_obs)
        if 'optical_flow' in self.modalities:
            vision_obs['optical_flow'] = self.get_optical_flow(raw_vision_obs)
        if 'scene_flow' in self.modalities:
            vision_obs['scene_flow'] = self.get_scene_flow(raw_vision_obs)
        if 'normal' in self.modalities:
            vision_obs['normal'] = self.get_normal(raw_vision_obs)
        if 'seg' in self.modalities:
            vision_obs['seg'] = self.get_seg(raw_vision_obs)
        return vision_obs
