import yaml
import os
import logging
from scipy.spatial.transform import Rotation as R
import numpy as np

wd = os.getcwd()
root_path = os.path.join(wd[:wd.find('/openvrooms')], 'openvrooms') 
#print(root_path)
dataset_path = os.path.join(root_path, "dataset")
interative_dataset_path = os.path.join(dataset_path, "interactive")
metadata_path = os.path.join(interative_dataset_path, "metadata")
assets_path = os.path.join(root_path, "assets")
config_path = os.path.join(assets_path, "configs")	
turtlebot_urdf_file = os.path.join(assets_path, "models/turtlebot/turtlebot.urdf")
husky_urdf_file = os.path.join(assets_path, "models/husky/husky.urdf")

def get_scene_path(scene_id):
    assert scene_id in os.listdir(interative_dataset_path), print("Error: scene folder %s doesn't exit!"%(scene_id))

    return os.path.join(interative_dataset_path, scene_id)

