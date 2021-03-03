import yaml
import os
import logging
from scipy.spatial.transform import Rotation as R
import numpy as np

wd = os.getcwd()
root_path = os.path.join(wd[:wd.find('/openvrooms')], 'openvrooms') 
dataset_path = os.path.join(root_path, "dataset")
interative_dataset_path = os.path.join(dataset_path, "interactive")
metadata_path = os.path.join(interative_dataset_path, "metadata")
assets_path = os.path.join(root_path, "assets")

code_path = os.path.join(root_path, "openvrooms")
config_path = os.path.join(code_path, "configs")	

original_dataset_path = "/dataset/openrooms/original"
brdf_dataset_path = os.path.join(original_dataset_path, "BRDFOriginDataset")

turtlebot_urdf_file = os.path.join(assets_path, "models/turtlebot/turtlebot.urdf")


def get_scene_path(scene_id):
    assert scene_id in os.listdir(interative_dataset_path), print("Error: scene folder %s doesn't exit!"%(scene_id))

    return os.path.join(interative_dataset_path, scene_id)

