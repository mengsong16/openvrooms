#!/usr/bin/env python

import os
import sys
import time
import random
import gibson2
import argparse

from gibson2.simulator import Simulator
from gibson2.utils.map_utils import gen_trav_map

from openvrooms.scenes.room_scene import RoomScene
from openvrooms.config import *


def generate_trav_map(scene_id='scene0420_01'):
    scene = RoomScene(scene_id=scene_id, fix_interactive_objects=True)
    
    s = Simulator(mode='headless', image_width=512,
                  image_height=512, device_idx=0)
    
    s.import_scene(scene)

    for i in range(480):
        s.step()
   
    vertices_info, faces_info = s.renderer.dump()
    s.disconnect()


    trav_map_filename_format = 'floor_trav_{}.png'
    obstacle_map_filename_format = 'floor_{}.png'
    

    gen_trav_map(vertices_info, faces_info, 
                 output_folder=get_scene_path(scene_id),
        trav_map_filename_format = trav_map_filename_format,
        obstacle_map_filename_format =obstacle_map_filename_format)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Traversability Map')
    parser.add_argument("--id", default='scene0420_01', help="Scene ID")
    args = parser.parse_args()

    generate_trav_map(scene_id=args.id)
