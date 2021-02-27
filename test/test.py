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
from openvrooms.objects.interactive_object import InteractiveObj

def test_object():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)
    
    scene_path = get_scene_path(scene_id='scene0420_01')
    chair1 = os.path.join(scene_path, '03001627_9231ef07326eae09b04cb542e2c50eb4_object_alignedNew.urdf')
    chair2 = os.path.join(scene_path, '03001627_fe57bad06e1f6dd9a9fe51c710ac111b_object_alignedNew.urdf')
    curtain = os.path.join(scene_path, 'curtain_4_object_alignedNew.urdf')

    obj1 = InteractiveObj(curtain)
    #obj1 = InteractiveObj(filename=curtain)
    obj1.load()
    #obj1.set_position([0,0,0.5])

    for _ in range(240000):  # at least 100 seconds
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()


def test_layout():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    #floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    #p.loadMJCF(floor)

    scene = RoomScene(scene_id='scene0420_01', fix_interactive_objects=True)
    scene.load_scene_metainfo()
    scene.load_layout()


    for _ in range(240000):  # at least 100 seconds
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()

def test_relocate_scene(scene_id='scene0420_01', n_interactive_objects=1):
    time_step = 1./240. 
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(time_step)
    
    scene = RelocateScene(scene_id=scene_id, n_interactive_objects=n_interactive_objects)
    scene.load()
    
    robot_config = parse_config(os.path.join(config_path, "turtlebot_relocate.yaml"))
    turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file) 

    turtlebot.load()
    
    turtlebot.set_position([0, 0, 0])
    turtlebot.robot_specific_reset()
    turtlebot.keep_still()
    
    for _ in range(2400000):  # at least 100 seconds
         p.stepSimulation()
         time.sleep(1./240.)

    p.disconnect()

def test_navigate_scene(scene_id='scene0420_01', n_obstacles=1):
    time_step = 1./240. 
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(time_step)
    
    scene = NavigateScene(scene_id=scene_id, n_obstacles=n_obstacles)
    scene.load()
    
    robot_config = parse_config(os.path.join(config_path, "turtlebot_navigate.yaml"))
    turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file) 

    turtlebot.load()
    
    turtlebot.set_position([0, 0, 0])
    turtlebot.robot_specific_reset()
    turtlebot.keep_still()
    
    for _ in range(2400000):  # at least 100 seconds
         p.stepSimulation()
         time.sleep(1./240.)

    p.disconnect()

def test_scene(scene_id='scene0420_01', fix_interactive_objects=True):
    time_step = 1./240. 
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(time_step)

    #floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    #p.loadMJCF(floor)
    scene = RoomScene(scene_id=scene_id, load_from_xml=True, fix_interactive_objects=fix_interactive_objects, empty_room=False)
    scene.load()
    
    #scene.change_interactive_objects_dynamics(mass=100)
    
    robot_config = parse_config(os.path.join(config_path, "turtlebot_interactive_demo.yaml"))
    turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file) 

    turtlebot.load()
    
    turtlebot.set_position([0, 0, 0])
    turtlebot.robot_specific_reset()
    turtlebot.keep_still()
    
    for _ in range(2400000):  # at least 100 seconds
         p.stepSimulation()
         time.sleep(1./240.)

    p.disconnect()

def test_robot(robot_name='turtlebot'):
    p.connect(p.GUI)
    #p.connect(p.DIRECT)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    if robot_name == 'turtlebot':
        config = parse_config(os.path.join(config_path, "turtlebot_navigate.yaml"))
        robot = Turtlebot(config=config) 
    else:
        config = parse_config(os.path.join(config_path, "fetch_navigate.yaml"))
        robot = Fetch(config=config)    

    robot.load()
    robot.set_position([0, 0, 0])
    robot.robot_specific_reset()
    robot.keep_still() 

    #print(turtlebot.get_position())
    #print(turtlebot.get_orientation())

    #print(len(robot.ordered_joints))
    #print(robot.control)
    #for n, j in enumerate(robot.ordered_joints):
    #    print(j.joint_name)

    
    for _ in range(2400000):  # move with small random actions for 10 seconds
        #action = np.random.uniform(-1, 1, robot.action_dim)
        action = random.randint(0, robot.action_space.n-1)
        robot.apply_action(action)
        p.stepSimulation()
        #time.sleep(1./240.0)
    
    p.disconnect()

def load_still_robot(robot, pos=[0,0,0]):
    robot.load()
    robot.set_position(pos)
    robot.robot_specific_reset()
    robot.keep_still()

def test_various_robot(scene_id):
    time_step = 1./240. 
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(time_step)
    
    scene = NavigateScene(scene_id=scene_id, n_obstacles=1)
    scene.load()
    
    robot_config = parse_config(os.path.join(config_path, "seudo_config.yaml"))

    # turtlebot
    turtlebot = Turtlebot(config=robot_config) 
    load_still_robot(turtlebot, pos=[0, 0, 0])

    # JR2
    jr2 = JR2(config=robot_config)
    #load_still_robot(jr2, pos=[1, 0, 0])

    # Kinova
    kinova = JR2_Kinova(config=robot_config)
    #load_still_robot(kinova, pos=[0, -1, 0])

    # Fetch
    fetch = Fetch(config=robot_config)
    load_still_robot(fetch, pos=[0, -1, 0])

    # Freight
    freight = Freight(config=robot_config)
    #load_still_robot(freight, pos=[0, 1, 0])

    # Locobot
    locobot = Locobot(config=robot_config)
    load_still_robot(locobot, pos=[0, 1, 0])


    # simulation    
    for _ in range(2400000):  # at least 100 seconds
         p.stepSimulation()
         time.sleep(1./240.)

    p.disconnect()


class DemoInteractive(object):
    def __init__(self):
        return

    def run_demo(self):
        s = Simulator(mode='pbgui', image_width=700, image_height=700)

        robot_config = parse_config(os.path.join(config_path, "turtlebot_interactive_demo.yaml"))
        turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file)

        scene = RoomScene(scene_id='scene0420_01', fix_interactive_objects=True)

        s.import_scene(scene)
        s.import_robot(turtlebot)
        
        time_step = 1./350.
        
        for i in range(10000):
            turtlebot.apply_action([0.1,0.5])
            s.step()
        
        s.disconnect()

def test_igibson_floor():
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

def floor_collision_detection(robot_id, floor_id):
    collision_links = list(p.getContactPoints(bodyA=robot_id, bodyB=floor_id))
    
    return len(collision_links) > 0 

def test_igibson_floor_robot_collision():
    config_filename = os.path.join(
        gibson2.root_path, 'test', 'test_house.yaml')
    env = iGibsonEnv(config_file=config_filename, mode='headless')

    floor_id = env.simulator.scene.floor_body_ids[0]
    robot_id = env.robots[0].robot_ids[0]

    print(floor_id)
    print(robot_id)

    print(str(env.collision_ignore_body_b_ids))

    floor_collision_steps = 0
    for j in range(2):
        env.reset()
        print('After reset')
        print(str(env.collision_ignore_body_b_ids))
        for i in range(300):
            action = env.action_space.sample()
            env.step(action)
            floor_collision_steps += floor_collision_detection(robot_id, floor_id)
            print('----------------------------------')
            print(env.collision_step)
            print(floor_collision_steps)
    
    env.close()

# move forward for a given world frame distance
def robot_move_forward(desired_distance, robot, max_steps):
    n_timestep = 0
    robot_distance = 0.0
    prev_position = robot.get_position()
    total_energy = 0.0

    for _ in range(max_steps):
        # move robot
        robot.apply_action(0)

        # one step simulation
        p.stepSimulation()

        # get normalized joint velocity and torque
        total_energy += robot.get_energy(normalized=True)

        # update cumulated distance
        current_position = robot.get_position()
        robot_distance += l2_distance(prev_position, current_position)
        prev_position = current_position
        
        n_timestep += 1

        if robot_distance >= desired_distance:
            break
    
    '''
    print("---------------------------")
    print('Desired distance: %f'%(desired_distance))
    print('Real distance: %f'%(robot_distance))
    print('Time steps: %d'%(n_timestep))
    print('Total total_energy: %f'%(total_energy))
    '''

    return robot_distance, n_timestep, total_energy

def push_forward(robot, obj, scene, target_position, unit_distance, time_step):
    robot_start_position = robot.get_position()
    object_start_position = obj.get_xy_position()

    totol_energy = 0.0
    step = 0
    while True:
    #for _ in range(30):
        robot_distance, n_timestep, step_energy = robot_move_forward(unit_distance, robot, 200)

        totol_energy += step_energy
        step += n_timestep

        # reach goal?    
        current_position = obj.get_xy_position()
        #dist = l2_distance(current_position, target_position)

        print('Steps: %d'%(step))
        print('Steps taken: %d'%(n_timestep))
        print('Robot traveled distance: %f'%(robot_distance))
        print('Energy cost: %f'%(step_energy))
        print("---------------------------")
        

        #if dist < 0.1:
        if current_position[0] > target_position[0]:
            break

    robot_end_position = robot.get_position()
    object_end_position = obj.get_xy_position()
    
    print("---------------------------")
    print("Object mass: %f"%(obj.get_mass()))
    print("Robot mass: %f"%(robot.get_mass()))
    print("Object FC: %f"%(obj.get_friction_coefficient()))
    print("Floor FC: %f"%(scene.get_floor_friction_coefficient()))
    print("---------------------------")
    print('Object start position: %s'%(object_start_position))
    print('Object end position: %s'%(object_end_position))
    print('Object target position: %s'%(target_position))
    print('Robot start position: %s'%(robot_start_position))
    print('Robot end position: %s'%(robot_end_position))
    print('Total energy: %f'%totol_energy)   
    print("---------------------------")  
    print("Robot wheel velocity (normalized): %f"%(robot.wheel_velocity))
    print("Physics timesteps: %f"%(time_step))
    print("Total steps: %d"%(step))
    print("---------------------------") 


def test_robot_energy_cost(scene_id='scene0420_01', n_interactive_objects=1):
    #time_step = 1./240. 
    time_step = 1./40.
    p.connect(p.GUI)
    #p.connect(p.DIRECT)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(time_step)
    
    #scene = NavigateScene(scene_id=scene_id, n_obstacles=n_interactive_objects)
    scene = RelocateScene(scene_id=scene_id, n_interactive_objects=n_interactive_objects)
    scene.load()

    obj = scene.interative_objects[0]
 
    
    
    #robot_name = 'turtlebot'
    robot_name = 'fetch'

    if robot_name == 'turtlebot':
        #scene.set_floor_friction_coefficient(mu=0.314287)
        #scene.set_floor_friction_coefficient(mu=0.764784) 
        obj.set_xy_position(-1.5, 0)
        #obj.set_mass(0.5)
        robot_config = parse_config(os.path.join(config_path, "turtlebot_relocate.yaml"))
        robot = Turtlebot(config=robot_config) 
    else:
        obj.set_xy_position(-1.4, 0)
        obj.set_mass(10)
        #scene.set_floor_friction_coefficient(mu=0.4)
        #scene.set_floor_friction_coefficient(mu=0.764784)

        robot_config = parse_config(os.path.join(config_path, "fetch_relocate.yaml"))
        robot = Fetch(config=robot_config)    

    robot.load()
    
    robot.set_position([-2, 0, 0])
    robot.robot_specific_reset()
    robot.keep_still()

    
    if robot_name == 'turtlebot':
        push_forward(robot, obj, scene, target_position=[2.5, 0], unit_distance = 0.1, time_step=time_step)
    else:
        push_forward(robot, obj, scene, target_position=[1, 0], unit_distance = 0.1, time_step=time_step)


    '''
    for _ in range(2400000):  # at least 100 seconds
        action = random.randint(0, robot.action_space.n-1)
        robot.apply_action(action)
        p.stepSimulation()

        #print()
        robot.get_energy(normalized=True)
        #print('--------------------------------')

        #time.sleep(time_step)
    '''

    #robot_move_forward(desired_distance=2, robot=robot, max_steps=2000)
    #robot_move_forward(desired_distance=2, robot=robot, max_steps=1000)

    # cool down
    robot.keep_still()
    for _ in range(2400000):
        p.stepSimulation()
        time.sleep(time_step)

    p.disconnect()


if __name__ == "__main__":
    aparser = argparse.ArgumentParser(description="Run run_demo.")
    aparser.add_argument("--id", default='scene0420_01', help="Scene ID")
    args = aparser.parse_args()

    test_robot_energy_cost()
    
    #test_relocate_scene(args.id, n_interactive_objects=1)
    #test_navigate_scene(args.id, n_obstacles=1)
    #test_scene(args.id, fix_interactive_objects=False)
    #test_layout()
    #test_robot(robot_name='fetch')
    #test_robot(robot_name='turtlebot')
    #test_object()
    #test_various_robot(args.id)

    '''
    demo = DemoInteractive()
    demo.run_demo()
    '''
    #test_igibson_floor()
    #test_igibson_floor_robot_collision()