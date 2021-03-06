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
import shutil

from gibson2.utils.utils import parse_config

# parse and generate mtl for one scene
def parse_generate_mtl_scene(scene_id, multi_band=False, borders=[-1., 0.], border_type="y_border", is_box_replaced=False, reverse_two_band=False, small_box=True):
	#original_dataset_path = os.path.join(dataset_path, 'original')
	kwargs = {
		#'scene_root': os.path.join(original_dataset_path, 'scenes'),
		'scene_root': dataset_path,
		'brdf_root' : os.path.join(original_dataset_path, 'BRDFOriginDataset'),
		'uvMapped_root': os.path.join(original_dataset_path, 'uv_mapped'),
		'envDataset_root': os.path.join(original_dataset_path, 'EnvDataset'),
		'layoutMesh_root': os.path.join(original_dataset_path, 'layoutMesh')
	}

	if multi_band:
		if reverse_two_band:
			suffix = 'multi_band_reverse'
		else:
			suffix = 'multi_band'	
		split_floor = True
	else:
		suffix = None
		split_floor = False

	# parse the scene and generate mtl files
	parser = SceneParser(scene_id=scene_id, save_root=interative_dataset_path, suffix=suffix, **kwargs)
	if multi_band:
		parser.parse(split_floor=split_floor, is_box_replaced=is_box_replaced, borders=borders, border_type=border_type, reverse_two_band=reverse_two_band, small_box=small_box)
	else:
		parser.parse(split_floor=split_floor, is_box_replaced=is_box_replaced, small_box=small_box)	

	## object list of the scene to a pickle file
	pickle_path = get_pickle_path(scene_id, suffix=suffix)
	parser.save_param(pickle_path)

# generate urdf for one scene
def generate_urdf_scene(scene_id, multi_band=False, reverse_two_band=False):
	# load object list
	parser = SceneParser(scene_id=scene_id)
	if multi_band:
		if reverse_two_band:
			suffix = 'multi_band_reverse'
		else:
			suffix = 'multi_band'
	else:
		suffix = None

	pickle_path = get_pickle_path(scene_id, suffix=suffix)
	parser.load_param(pickle_path)
	parser.print_param()

	urdf_prototype_file = os.path.join(metadata_path, 'urdf_prototype.urdf') # urdf template

	scene_folder = get_scene_path(scene_id, suffix=suffix)
	if not os.path.exists(scene_folder):
		os.makedirs(scene_folder)

	
	for obj in parser.obj_list:
		obj_file = os.path.join(scene_folder, obj.obj_path)	
		#print(obj_file)

		# floor has no center of mass
		if 'floor' in obj_file:
			# copy .obj to vhach.obj
			floor_vhacd_obj_path = os.path.join(scene_folder, obj_file.replace('.obj', '_vhacd.obj'))
			shutil.copyfile(obj_file, floor_vhacd_obj_path)	
			print('Generated floor vhacd obj file.')

			generate_urdf(obj_file, scene_folder, urdf_prototype_file, decompose_concave=True, force_decompose=False, mass=100, center=None)
			#builder.build_urdf(filename=floor_obj_path, force_overwrite=True, decompose_concave=True, force_decompose=False, mass=100, center=None)
			print('Generated floor urdf file.')
		else:
			#continue
			generate_urdf(obj_file, scene_folder, urdf_prototype_file, decompose_concave=True, force_decompose=True, center='geometric')	

	print('-------------------------------------')
	print('URDF generation Done.')	
	print('%d files in total.'%len(parser.obj_list))
	print('-------------------------------------')
	
# parse, generate mtl and urdf for all scenes
def data_generation_all_scenes():
	for scene_id in scene_list:
		parse_generate_mtl_scene(scene_id)
		generate_urdf_scene(scene_id)

def data_generation_one_scene(scene_id, multi_band, is_box_replaced, reverse_two_band, small_box):
	if multi_band == True:
		config_file = os.path.join(config_path, 'fetch_relocate_two_band_short.yaml')
		config = parse_config(config_file)
		parse_generate_mtl_scene(scene_id, multi_band=multi_band, borders=np.array(config.get('floor_borders')), border_type=config.get('border_type'), is_box_replaced=is_box_replaced, reverse_two_band=reverse_two_band, small_box=small_box)
		generate_urdf_scene(scene_id, multi_band=multi_band, reverse_two_band=reverse_two_band)
	else:
		parse_generate_mtl_scene(scene_id, multi_band=multi_band, is_box_replaced=is_box_replaced, reverse_two_band=reverse_two_band, small_box=small_box)
		generate_urdf_scene(scene_id, multi_band=multi_band, reverse_two_band=reverse_two_band)	
	


if __name__ == '__main__':
	aparser = argparse.ArgumentParser(description="Run data preprocessing.")
	aparser.add_argument("--id", default='scene0420_01', help="Scene ID")
	args = aparser.parse_args()

	# 2 band, replace with clean boxes, small box
	#data_generation_one_scene(scene_id=args.id, multi_band=True, is_box_replaced=True, reverse_two_band=False, small_box=True)

	# no band, replace with clean boxes, big box
	data_generation_one_scene(scene_id=args.id, multi_band=False, is_box_replaced=True, reverse_two_band=False, small_box=False)
	
