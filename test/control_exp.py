import yaml
from openvrooms.robots.turtlebot import Turtlebot
from openvrooms.robots.fetch_robot import Fetch

from gibson2.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2.utils.utils import l2_distance
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.profiler import Profiler
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.scene_base import Scene

#from gibson2.robots.husky_robot import Husky
#from gibson2.robots.ant_robot import Ant
#from gibson2.robots.humanoid_robot import Humanoid
from gibson2.robots.jr2_robot import JR2
from gibson2.robots.jr2_kinova_robot import JR2_Kinova
#from gibson2.robots.quadrotor_robot import Quadrotor
#from gibson2.robots.fetch_robot import Fetch
from gibson2.robots.freight_robot import Freight
from gibson2.robots.locobot_robot import Locobot

import pytest
import pybullet as p
import numpy as np
import os
import gibson2

from gibson2.envs.igibson_env import iGibsonEnv


import sys
#sys.path.insert(0, "../")
from openvrooms.config import *

import time

import pybullet_data
import cv2

from scipy.spatial.transform import Rotation as R

import trimesh

import argparse
import random

from openvrooms.scenes.room_scene import RoomScene
from openvrooms.scenes.relocate_scene import RelocateScene
from openvrooms.scenes.navigate_scene import NavigateScene
from openvrooms.scenes.relocate_scene_different_objects import RelocateSceneDifferentObjects
from openvrooms.objects.interactive_object import InteractiveObj

from random import randrange

from openvrooms.envs.relocate_env import RelocateEnv
from openvrooms.envs.navigate_env import NavigateEnv

from gibson2.utils.utils import parse_config


def test_robot_energy_cost_agent_level(mode="gui", exp_type="push_forward"):
    config_file_path = os.path.join(config_path, "controlled_pushing_exp.yaml")
    if exp_type == "push_forward": 
        env = RelocateEnv(config_file=config_file_path, mode=mode)
    else:
        env = NavigateEnv(config_file=config_file_path, mode=mode)

    env.reset()

    # warm up
    for _ in range(200):
        env.step(3)
    
    # experiment (push, random ...)
    simple_trajectory_agent_level(env=env, config_file=config_file_path, exp_type=exp_type)

    # cool down
    for _ in range(2400000):
        env.step(3)

    env.close()


def push_forward(env, scene, robot, obj, config):
    object_start_position = env.task.obj_initial_pos[0]
    object_goal_position = env.task.obj_target_pos[0]
    robot_start_position = env.task.agent_initial_pos
    robot_energy_normalized = env.normalized_energy

    # action step, not simulation step
    step_num = 0

    while True:
        # move robot forward for one action step
        state, reward, done, info = env.step(0)

        step_num += 1

        # reach goal?    
        object_current_position = obj.get_xy_position()
        if object_current_position[0] > object_goal_position[0]:
            break

    robot_end_position = robot.get_position()
    object_end_position = obj.get_xy_position() 
    
    print("*******************************************************")
    print("Experiment summary: move forward")
    print("*******************************************************")
    print("Object mass: %f"%(obj.get_mass()))
    print("Robot mass: %f"%(robot.get_mass()))
    print("Object friction coefficient: %f"%(obj.get_friction_coefficient()))
    print("Floor friction coefficient: %f"%(scene.get_floor_friction_coefficient()))
    print("---------------------------")
    print("Robot wheel velocity (normalized): %f"%(robot.wheel_velocity)) # set in config
    print("Physics simulation timestep: %f"%(env.physics_timestep)) # set in config
    print("Action timestep: %f"%(env.action_timestep)) # set in config
    print("---------------------------")
    print('Object start position: %s'%(object_start_position))
    print('Object end position: %s'%(object_end_position))
    print('Object target position: %s'%(object_goal_position))
    print('Object traveled distance: %f'%(l2_distance(object_start_position, object_end_position)))
    print("---------------------------")
    print('Robot start position: %s'%(robot_start_position))
    print('Robot end position: %s'%(robot_end_position))
    print('Robot traveled distance: %f'%(l2_distance(robot_start_position, robot_end_position)))
    print("---------------------------")
    print("Total (action) steps: %d"%(step_num))
    if robot_energy_normalized:
        robot_energy_string = "Robot energy(normalized)"
    else:
        robot_energy_string = "Robot energy(raw)"    
    print(robot_energy_string+': episode: %f, per step: %f'%(env.current_episode_robot_energy_cost, env.current_episode_robot_energy_cost/float(step_num)))
    print('Pushing translation energy: episode: %f, per step: %f'%(env.current_episode_pushing_energy_translation, env.current_episode_pushing_energy_translation/float(step_num)))  
    print('Pushing rotation energy: episode: %f, per step: %f'%(env.current_episode_pushing_energy_rotation, env.current_episode_pushing_energy_rotation/float(step_num)))  
    print("*******************************************************") 

def random_move(env, scene, robot, config):
    robot_start_position = env.task.agent_initial_pos
    robot_energy_normalized = env.normalized_energy

    # action step, not simulation step
    step_num = 14

    for _ in range(step_num):  # 10 seconds
        action = env.action_space.sample()
        # move robot forward for one action step
        state, reward, done, info = env.step(action)
    
    print("*******************************************************")
    print("Experiment summary: random move")
    print("*******************************************************")
    print("Robot mass: %f"%(robot.get_mass()))
    print("Floor friction coefficient: %f"%(scene.get_floor_friction_coefficient()))
    print("---------------------------")
    print("Robot wheel velocity (normalized): %f"%(robot.wheel_velocity)) # set in config
    print("Physics simulation timestep: %f"%(env.physics_timestep)) # set in config
    print("Action timestep: %f"%(env.action_timestep)) # set in config
    print("---------------------------")
    print("Total (action) steps: %d"%(step_num))
    if robot_energy_normalized:
        robot_energy_string = "Robot energy(normalized)"
    else:
        robot_energy_string = "Robot energy(raw)"    
    print(robot_energy_string+': episode: %f, per step: %f'%(env.current_episode_robot_energy_cost, env.current_episode_robot_energy_cost/float(step_num)))
    print("*******************************************************")        

def simple_trajectory_agent_level(env, config_file, exp_type="push_forward"):
    scene = env.scene
    robot = env.robots[0]

    config = parse_config(config_file)
    
    # push forward for some distance
    if exp_type == "push_forward":
        obj = env.scene.interative_objects[0] 
        push_forward(env, scene, robot, obj, config)
    # random move for some steps            
    elif exp_type == "random_move": 
        random_move(env, scene, robot, config) 
    else:
        print("Error: undefined experiment type!!")   

if __name__ == "__main__":
    test_robot_energy_cost_agent_level(mode="gui", exp_type="random_move")


