import sys
#sys.path.insert(0, "../")
#from config import *
from openvrooms.config import *

import os
import numpy as np
import pickle
from openvrooms.data_processor.xml_parser import SceneParser
from openvrooms.data_processor.urdf_generator import generate_urdf

import argparse

# parse and generate mtl for one scene
def parse_generate_mtl_scene(scene_id):
	original_dataset_path = os.path.join(dataset_path, 'original')
	kwargs = {
		'scene_root': os.path.join(original_dataset_path, 'scenes'),
		'brdf_root' : os.path.join(original_dataset_path, 'BRDFOriginDataset'),
		'uvMapped_root': os.path.join(original_dataset_path, 'uv_mapped'),
		'envDataset_root': os.path.join(original_dataset_path, 'EnvDataset'),
		'layoutMesh_root': os.path.join(original_dataset_path, 'layoutMesh')
	}

	# parse the scene and generate mtl files
	parser = SceneParser(scene_id=scene_id, save_root=interative_dataset_path, **kwargs)
	parser.parse()

	## object list of the scene to a pickle file
	pickle_path = os.path.join(metadata_path, str(scene_id)+'.pkl')
	parser.save_param(pickle_path)

# generate urdf for one scene
def generate_urdf_scene(scene_id):
	# load object list
	parser = SceneParser(scene_id=scene_id)
	pickle_path = os.path.join(metadata_path, str(scene_id)+'.pkl')
	parser.load_param(pickle_path)
	parser.print_param()

	urdf_prototype_file = os.path.join(metadata_path, 'urdf_prototype.urdf') # urdf template
	scene_folder = get_scene_path(scene_id)

	
	for obj in parser.obj_list:
		obj_file = os.path.join(scene_folder, obj.obj_path)	
		print(obj_file)
		generate_urdf(obj_file, scene_folder, urdf_prototype_file, force_decompose=True, center='geometric')

	print('-------------------------------------')
	print('URDF generation Done.')	
	print('%d files in total.'%len(parser.obj_list))
	print('-------------------------------------')
	
# parse, generate mtl and urdf for all scenes
def data_generation_all_scenes():
	for scene_id in scene_list:
		parse_generate_mtl_scene(scene_id)
		generate_urdf_scene(scene_id)

if __name__ == '__main__':
	aparser = argparse.ArgumentParser(description="Run data preprocessing.")
	aparser.add_argument("--id", default='scene0420_01', help="Scene ID")
	args = aparser.parse_args()
	
	parse_generate_mtl_scene(args.id)
	generate_urdf_scene(args.id)
	
	
