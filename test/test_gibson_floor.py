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

if __name__ == "__main__":
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
		