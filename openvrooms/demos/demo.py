import yaml
from openvrooms.robots.turtlebot import Turtlebot
from gibson2.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2.utils.utils import l2_distance
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.profiler import Profiler

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
from openvrooms.objects.interactive_object import InteractiveObj


def pushing_demo(scene_id='scene0420_01'):
    time_step = 1./240.
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(time_step)

    # load scene and set desk's position
    scene = RoomScene(scene_id=scene_id, load_from_xml=False, fix_interactive_objects=False)
    scene.load()
    obj = scene.interative_objects[0]
    obj.set_xy_position(0.5, 0)
    obj_id = obj.body_id
    obj_mesh = obj.get_mesh()
    # bounds - axis aligned bounds of mesh
    # 2*3 matrix, min, max, x, y, z
    obj_bounds = obj_mesh.bounds
    obj_com = obj.get_mesh_com()
    # transform from mesh frame to COM link frame
    obj_left_bound = np.array([obj_bounds[0][0]-obj_com[0], 0, 0])

    # set floor friction coefficient
    if 'carpet_loop' in scene_id:
        scene.set_floor_friction_coefficient(mu=0.764784) 
    else:
        scene.set_floor_friction_coefficient(mu=0.314287)

    obj.set_mass(2)
    obj.set_friction_coefficient(0.764465) 


    # load robot and set its position
    robot_config = parse_config(os.path.join(config_path, "turtlebot_interactive_demo.yaml"))
    turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file) 
    turtlebot.load()
    
    turtlebot.set_position([0, 0, 0])
    turtlebot.robot_specific_reset()
    turtlebot.keep_still()
    
    # warm up
    warm_up_steps = 1200
    for _ in range(warm_up_steps):
        p.stepSimulation()
        time.sleep(time_step)
    

    # ------------- experiment START ------------- 
    force_steps = 20
    observe_steps = 400

    obj_start_pos = obj.get_position()
    robot_start_pos = turtlebot.get_position()

    # apply an external force on the object for some time
    for _ in range(force_steps):
        p.stepSimulation()
        #p.applyExternalForce(objectUniqueId=obj_id, linkIndex=-1, forceObj=[10,0,0], posObj=obj_left_bound, flags=p.LINK_FRAME)
        turtlebot.move_forward(0.01)
        time.sleep(time_step)
    
    # no external force for some time
    for _ in range(observe_steps):  # at least 100 seconds
         p.stepSimulation()
         time.sleep(time_step)

    # --------------- experiment END ----------------
    turtlebot.keep_still()
    # cool down
    warm_up_steps = 2400
    for _ in range(warm_up_steps):
        p.stepSimulation()
        time.sleep(time_step)


    robot_end_pos = turtlebot.get_position()
    obj_end_pos = obj.get_position()
    # compute distance robot moved
    robot_distance = l2_distance(robot_start_pos, robot_end_pos)
    # compute distance object moved
    obj_distance = l2_distance(obj_start_pos, obj_end_pos)
    # compute time
    force_time = time_step * force_steps
    observe_time = time_step * observe_steps
    simulation_time = force_time + observe_time

    print("*---------------------------*")
    print("Scene id: %s"%(scene_id))
    #print("Object left bound (COM link frame): %s"%(str(obj_left_bound)))
    print("Object mass (kg): %f"%(obj.get_mass()))
    print("Robot mass (kg): %f"%(turtlebot.get_mass()))
    print("Object friction coefficient: %f"%(obj.get_friction_coefficient()))
    print("Floor friction coefficient: %f"%(scene.get_floor_friction_coefficient()))
    print("---------------------------")
    print("Object moving distance (m): %f"%(obj_distance))
    print("Robot moving distance (m): %f"%(robot_distance))
    print("Force exerting time (s): %f"%(force_time))  
    print("No force time (s): %f"%(observe_time))
    print("Total simulation time (s): %f"%(simulation_time))
    print("*---------------------------*")   

    p.disconnect()



# move forward for a given world frame distance
def move_forward(distance, robot, max_steps, time_step=1./240.):
    n_timestep = 0
    robot_distance = 0.0
    prev_position = robot.get_position()
    total_energy = 0.0
    for _ in range(max_steps):
        # move robot
        robot.apply_action(0)
        # one step simulation
        p.stepSimulation()
        time.sleep(time_step)

        # get normalized joint velocity and torque
        joint_velocity, joint_torque = robot.get_joint_info()
        electricity_cost = float(np.abs(np.array(joint_velocity)*np.array(joint_torque)).mean())
        stall_torque_cost = float(np.square(joint_torque).mean())
        total_energy += (electricity_cost+stall_torque_cost)

        # update cumulated distance
        current_position = robot.get_position()
        robot_distance += l2_distance(prev_position, current_position)
        prev_position = current_position
        
        n_timestep += 1

        if robot_distance >= distance:
            break

    return robot_distance, n_timestep, total_energy    

def navigation_demo():
    time_step = 1./240.
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(time_step)

    # load scene
    scene = RoomScene(scene_id='scene0420_01', load_from_xml=False, fix_interactive_objects=False, empty_room=True)
    scene.load()

    # set floor friction coefficient
    scene.set_floor_friction_coefficient(mu=0.1)

    # load robot and set its position
    robot_config = parse_config(os.path.join(config_path, "turtlebot_interactive_demo.yaml"))
    turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file)
    # set max joint velocity
    turtlebot.set_velocity(1)
    # set discrete or continuous
    turtlebot.is_discrete = True  
    turtlebot.load()

    turtlebot.set_position([0, 0, 0])
    turtlebot.robot_specific_reset()
    turtlebot.keep_still()

    # warm up
    warm_up_steps = 240
    for _ in range(warm_up_steps):
        p.stepSimulation()
        time.sleep(time_step)

    # ------------- experiment START ------------- 
    total_distance, n_timestep, total_energy = move_forward(1, turtlebot, max_steps=1000, time_step=time_step)

    # --------------- experiment END ----------------
    turtlebot.keep_still()

    # print stats
    print("*---------------------------*")
    print("Floor friction coefficient: %f"%(scene.get_floor_friction_coefficient()))
    print("Distance: %f"%(total_distance))
    print("Time steps: %d"%(n_timestep))
    print("Total energy: %f"%(total_energy))
    print("*---------------------------*")
    

def navigation_demo_reset_positions(scene_id='scene0420_01'):
    time_step = 1./240. 
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(time_step)

   
    # load scene
    scene = RoomScene(scene_id=scene_id, load_from_xml=True, fix_interactive_objects=True)
    scene.load()
    
    
    # load robot
    robot_config = parse_config(os.path.join(config_path, "turtlebot_interactive_demo.yaml"))
    turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file) 
    turtlebot.load()
    
    
    turtlebot.set_position([-1.5, 2, 0])
    turtlebot.robot_specific_reset()
    turtlebot.keep_still() 
    
    # warm up
    warm_up_steps = 1200
    for _ in range(warm_up_steps):
        p.stepSimulation()
        time.sleep(time_step)

    # ------------- experiment START ------------- 
    # move forward for 1.5m
    for _ in range(150):
        turtlebot.move_forward(0.01)
        p.stepSimulation()
        time.sleep(time_step)

    # turn right 90 degrees:
    for _ in range(10):
        turtlebot.turn_right(0.157)
        p.stepSimulation()
        time.sleep(time_step)

    # move forward for 3.4m
    for _ in range(170):
        turtlebot.move_forward(0.02)
        p.stepSimulation()
        time.sleep(time_step)

    # turn left 45 degrees：
    for _ in range(5):
        turtlebot.turn_left(0.157)
        p.stepSimulation()
        time.sleep(time_step)

    # move forward for 1m
    for _ in range(100):
        turtlebot.move_forward(0.01)
        p.stepSimulation()
        time.sleep(time_step)

    # turn left 45 degrees：
    for _ in range(5):
        turtlebot.turn_left(0.157)
        p.stepSimulation()
        time.sleep(time_step)

    # move forward for 0.8m
    for _ in range(80):
        turtlebot.move_forward(0.01)
        p.stepSimulation()
        time.sleep(time_step)
    
    # --------------- experiment END ----------------
    turtlebot.keep_still()

    # cool down
    warm_up_steps = 2400
    for _ in range(warm_up_steps):
        p.stepSimulation()
        time.sleep(time_step)

    p.disconnect()


if __name__ == "__main__":
    aparser = argparse.ArgumentParser(description="Run run_demo.")
    aparser.add_argument("--id", default='scene0420_01', help="Scene ID")
    args = aparser.parse_args()

    # pushing_demo(args.id)
    navigation_demo()
    #navigation_demo_reset_positions(args.id)



