import xml.etree.ElementTree as ET
from shutil import copyfile, rmtree
import os
import numpy as np
import pickle
from openvrooms.data_processor.obj_transform import ObjTransform, ObjTransformBasic
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from math import ceil, floor
from openvrooms.data_processor.split import split_layout
from openvrooms.data_processor.floor_bbox import FloorBBox
from openvrooms.data_processor.box_bbox import BoxBBox
from openvrooms.data_processor.duplicate import duplicate_floor


class SceneObj(object):
	def __init__(self, **kwargs):
		self.__dict__.update(kwargs)

class SceneParser:
	static_objs = {'door', 'window', 'curtain', 'ceiling_lamp'}
	def __init__(self, scene_id: str, save_root=None, suffix=None, **kwargs):
		# add layout to static list whose name contains scene_id
		self.static_objs.add(scene_id)
		self.scene_id = scene_id
		self.obj_list = list()
		self.xml_root = None

		if save_root is not None:
			self.scene_root = kwargs['scene_root']
			self.brdf_root = kwargs['brdf_root']
			self.uvMapped_root = kwargs['uvMapped_root']
			self.envDataset_root = kwargs['envDataset_root']
			self.layoutMesh_root = kwargs['layoutMesh_root']
			if suffix is None:
				self.save_root = os.path.join(save_root, scene_id)
			else:
				self.save_root = os.path.join(save_root, scene_id+'_'+suffix) 

			if not os.path.exists(self.save_root):
				os.makedirs(self.save_root)       

	# get xml tree and save .xml to the output directory
	def __get_scene_xml(self, scene_xml_path):
		
		assert os.path.isfile(scene_xml_path), f"Error: file '{scene_xml_path}' doesn't exit!"
		
		copyfile(scene_xml_path, os.path.join(self.save_root, 'main.xml'))
		tree = ET.parse(scene_xml_path)
		
		return tree.getroot()
	
	def separate_static_interactive(self):
		interative_object_list = list()
		static_object_list = list()
		layout_list = list()

		for obj in self.obj_list:
			if self.scene_id in obj.id:
				layout_list.append(obj)
			elif obj.type == "static":
				static_object_list.append(obj)
			elif obj.type == "interactive":
				interative_object_list.append(obj)  
			else:
				print(f"[SceneParser.separate_static_interactive]Error: unknown object type:\nid={obj.id}\nobj_path={obj.obj_path}\ntype={obj.type}")  
		
		return static_object_list, interative_object_list, layout_list

	# get the transformation from obj reference frame to bullet reference frame
	def get_tranform_obj2bullet(self):
		r = R.from_euler('x', 90, degrees=True)
		quat = r.as_quat()

		return [('rotate', 'quaternion', np.array(quat))]

	# parse one shape block - one object instance
	def parse_shape_block(self, shape_block):
		#print(shape_block.get('id'))
		obj = SceneObj(id=shape_block.get('id'))

		## obj type: static or interactive
		obj.type = 'interactive'
		for static_obj in self.static_objs:
			if obj.id.lower().find(static_obj) != -1: # is a static obj
				obj.type = 'static'
				break

		## get pose
		obj.transforms = list()
		transform_block = shape_block.find('transform')
		for transform in transform_block:
			tag= transform.tag
			if tag == 'scale':
				obj.transforms.append(('scale', np.array([
					float(transform.get('x')),
					float(transform.get('y')),
					float(transform.get('z'))   
				])))
			elif tag == 'rotate':
				obj.transforms.append(('rotate', 'axisangle', np.array([
					float(transform.get('angle')),
					float(transform.get('x')),
					float(transform.get('y')),
					float(transform.get('z'))
				])))
			elif tag == 'translate':
				obj.transforms.append(('translate', np.array([
					float(transform.get('x')),
					float(transform.get('y')),
					float(transform.get('z'))
				])))

		## get .obj file path
		relative_path = shape_block.find('string').get('value')[15:]
		mega_type = relative_path.split('/')[0]
		if mega_type == 'layoutMesh':
			obj_path = os.path.join(self.layoutMesh_root, relative_path[len(mega_type)+1:])
		elif mega_type == 'uv_mapped':
			obj_path = os.path.join(self.uvMapped_root, relative_path[len(mega_type)+1:])
		else:
			obj_path = shape_block.find('string').get('value')
			assert os.path.isfile(obj_path), f"[SceneParser.parse_shape_block]Error: non-exist obj file: {obj_path}!"
		
		# new file .obj & .mtl paths
		file_name = relative_path.split('/')[-1]
		file_name = file_name[:-4]
		
		# remove container.obj
		if file_name == 'container': 
			return None

		# in case that different obj instances with different transformations share the same .obj file
		# name the transformed .obj files in increasing indices
		obj_idx = len([prev_obj for prev_obj in self.obj_list if prev_obj.id == obj.id])

		# set absolute path
		obj.obj_path = os.path.join(self.save_root, obj.id+('' if obj_idx==0 else str(obj_idx))+'.obj')
		obj.mtl_path = os.path.join(self.save_root, obj.id+'.mtl')

		# transform vertices & normals in .obj tile
		obj_trans  = ObjTransformBasic(obj_path)
		obj_trans.transform_vertices_and_normals(obj.transforms)
		# apply transformation from .obj frame to bullet frame
		obj_trans.transform_vertices_and_normals(self.get_tranform_obj2bullet())
		obj_trans.form_obj_file(overwrite=True)
		
		### create a mtl file
		shared_mtl_param_str = ''
		shared_mtl_param_str += 'Ns 1.190084\n'
		shared_mtl_param_str += 'Kd 0.800000 0.800000 0.800000\n'
		shared_mtl_param_str += 'Ka 1.000000 1.000000 1.000000\n'
		shared_mtl_param_str += 'Ks 0.500000 0.500000 0.500000\n'
		shared_mtl_param_str += 'Ke 0.000000 0.000000 0.000000\n'
		shared_mtl_param_str += 'Ni 1.450000\n'
		shared_mtl_param_str += 'd 1.000000\n'
		shared_mtl_param_str += 'illum 2\n'
		## write .mtl file 
		with open(obj.mtl_path, 'w') as f:
			f.write('# Material Count: 1\n\n')
			f.write('newmtl combined_material\n')
			f.write(shared_mtl_param_str)

		## bsdf -> mtl
		texture_size = 512
		# get the list of materials belonging to this object
		mtl_list = [e.get('id') for e in shape_block.findall('ref')]

		## combine textures
		# mtl2range = obj_trans.calc_tex_coord_range()
		mtl2uvScale = dict()
		combined_albedo_texture = list()
		combined_roughness_texture = list()
		# for each material name, find the corresponding bsdf block and parse it
		resize_scale = 1.
		for i, mtl_name in enumerate(mtl_list):
			# find the corresponding bsdf block
			for bsdf_block in self.xml_root.findall('bsdf'):
				if bsdf_block.get('id') == mtl_name: 
					break
			# bsdf block found, parse it    
			albedo_texture, roughness_texture, uvScale = self.parse_bsdf_block_get_texture(bsdf_block, texture_size)
			if (not isinstance(albedo_texture, np.ndarray)) and albedo_texture == None:
				tqdm.write(f"No texture for mtl {mtl_name}")
				mtl_list[i] = None
				continue
			obj_trans.apply_uvScale(mtl_name, uvScale)
			mtl2uvScale[mtl_name] = uvScale
			# combined_albedo_texture.append(albedo_texture)
			# combined_roughness_texture.append(roughness_texture)
			# replicate, crop, and resize the texture maps
			replicated_albedo_texture = list()
			replicated_roughness_texture = list()
			mtl2range = obj_trans.calc_tex_coord_range()
			((umin, vmin), (umax, vmax)) = mtl2range[mtl_name]
			# tqdm.write(f"mtl={mtl_name}|(umin, umax)=({umin}, {umax})|(vmin, vmax)=({vmin}, {vmax})")
			vmin_ceil, vmax_floor = ceil(vmin), floor(vmax)
			if i == 0: resize_scale = max(vmax_floor - vmin_ceil, 1)
			# replicate
			replicated_albedo_texture.append(np.tile(albedo_texture, (max(vmax_floor-vmin_ceil, 1), 1, 1)))
			replicated_roughness_texture.append(np.tile(roughness_texture, (max(vmax_floor-vmin_ceil, 1), 1, 1)))
			# crop
			replicated_albedo_texture = [ albedo_texture[-int(abs(vmax-vmax_floor)*texture_size):, :, :] ] + replicated_albedo_texture
			replicated_roughness_texture = [ roughness_texture[-int(abs(vmax-vmax_floor)*texture_size):, :, :] ] + replicated_roughness_texture
			replicated_albedo_texture.append( albedo_texture[:int(abs(vmin_ceil-vmin)*texture_size), :, :] )
			replicated_roughness_texture.append( roughness_texture[:int(abs(vmin_ceil-vmin)*texture_size), :, :] )
			# resize
			replicated_albedo_texture = np.concatenate(replicated_albedo_texture, axis=0)
			replicated_roughness_texture = np.concatenate(replicated_roughness_texture, axis=0)
			replicated_albedo_texture = cv2.resize(replicated_albedo_texture, dsize=(texture_size, int(texture_size*resize_scale)), interpolation=cv2.INTER_AREA)
			replicated_roughness_texture = cv2.resize(replicated_roughness_texture, dsize=(texture_size, int(texture_size*resize_scale)), interpolation=cv2.INTER_AREA)
			# store the replicated texture maps to the combined texture lists
			combined_albedo_texture.append(replicated_albedo_texture)
			combined_roughness_texture.append(replicated_roughness_texture)
		
		# filter out None
		mtl_list = list(filter(None.__ne__, mtl_list))
		if len(mtl_list) > 0:
			# tqdm.write(f"mtl_list: {mtl_list}")
			# concatenate textures
			combined_albedo_texture = np.concatenate(combined_albedo_texture, axis=0)
			combined_roughness_texture = np.concatenate(combined_roughness_texture, axis=0)
			# save combined texture images
			combined_albedo_texture_imgname = obj.id + '_diffuse_tiled_combined' + ('' if obj_idx==0 else str(obj_idx)) + '.png'
			combined_roughness_texture_imgname = obj.id + '_rough_tiled_combined' + ('' if obj_idx==0 else str(obj_idx)) + '.png'
			cv2.imwrite(os.path.join(self.save_root, combined_albedo_texture_imgname), combined_albedo_texture[:, :, ::-1])
			cv2.imwrite(os.path.join(self.save_root, combined_roughness_texture_imgname), combined_roughness_texture[:, :, ::-1])  
			
			## write .mtl file 
			with open(obj.mtl_path, 'a') as f:
				f.write('map_Kd ' + combined_albedo_texture_imgname + '\n')
				f.write('map_Pr ' + combined_roughness_texture_imgname + '\n')

		## change original texture coordinates in .obj to match the combined texture
		obj_trans.map_vt_to_combined_texture(mtl_list, texture_size, mtl2uvScale, overwrite=True)

		# ## remove vertex normals
		# obj_trans.remove_lines(startswith='vn ')

		# ## make faces two-sided
		# obj_trans.duplicate_faces()

		## save .obj to output directory
		obj_trans.save_obj_file(obj.obj_path)
		
		## add the filename of the .mtl file to .obj file
		with open(obj.obj_path, 'r+') as f:
			#content = f.read()
			old_lines = f.readlines()
			
			# write mtllib at the very beginning of the file
			f.seek(0, 0)
			f.write('mtllib '+obj.id+'.mtl\n')

			# write old lines and remove mtllib lines already been there
			for line in old_lines:
				if not 'mtllib' in line:
					f.write(line)

		## change obj.obj_path and obj.mtl_path to filename for save
		obj.obj_path = obj.obj_path.split('/')[-1]
		obj.mtl_path = obj.mtl_path.split('/')[-1]

		return obj

	def parse_bsdf_block_get_texture(self, bsdf_block, texture_size=512):
		# roughnessScale & uvScale
		roughnessScale = 1.
		uvScale = 1.
		for f in bsdf_block.findall('float'):
			if f.get('name') == 'roughnessScale': 
				roughnessScale = float(f.get('value'))
			elif f.get('name') == 'uvScale':
				uvScale = float(f.get('value'))
		# albedoScale
		albedoScale = np.ones((3, ), dtype=float)
		for rgb in bsdf_block.findall('rgb'):
			if rgb.get('name') == 'albedoScale':
				albedoScale = np.array([float(v) for v in rgb.get('value').strip().split()], dtype=float)
		# tqdm.write(f"{bsdf_block.get('id')}|albedoScale={albedoScale}|roughnessScale={roughnessScale}")
		# get texture maps
		albedo_texture = None
		roughness_texture = None
		for texture in bsdf_block.findall('texture'):
			# albedo / diffuse
			if texture.get('name') == 'albedo':
				albedo_path = os.path.join(self.brdf_root, texture.find('string').get('value')[33:])
				albedo_texture = cv2.imread(albedo_path).astype(np.uint8)[:texture_size, :texture_size, ::-1] # (texture_size, texture_size, 3) in RGB
				albedo_texture = (np.clip(((albedo_texture / 255.) ** (2.2) * albedoScale) ** (1./2.2), 0, 1) * 255).astype(np.uint8)
			# roughness
			if texture.get('name') == 'roughness':
				roughness_path = os.path.join(self.brdf_root, texture.find('string').get('value')[33:])
				roughness_texture = cv2.imread(roughness_path).astype(np.uint8)[:texture_size, :texture_size, ::-1] # (texture_size, texture_size, 3) in RGB
				roughness_texture = (np.clip((roughness_texture / 255.) * roughnessScale, 0, 1) * 255).astype(np.uint8)
				# roughness_texture *= roughnessScale
		if isinstance(albedo_texture, np.ndarray): assert albedo_texture.shape == (texture_size, texture_size, 3), f"[SceneParser.parse_bsdf_block_get_texture]Error: Invalid albedo texture shape {albedo_texture.shape}!"
		if isinstance(roughness_texture, np.ndarray): assert roughness_texture.shape == (texture_size, texture_size, 3), f"[SceneParser.parse_bsdf_block_get_texture]Error: Invalid roughness texture shape {roughness_texture.shape}!"
		return albedo_texture, roughness_texture, uvScale

	# parse one bsdf block
	def parse_bsdf_block(self, bsdf_block):
		mtl_param_str = ''
		# albedo
		rgb = bsdf_block.find('rgb')
		if rgb.get('name') == 'albedo': 
			mtl_param_str += 'Kd ' + rgb.get('value') + '\n'
		else:
			mtl_param_str += 'Kd 0.800000 0.800000 0.800000\n'  
		# roughness
		for f in bsdf_block.findall('float'):
			if f.get('name') == 'roughness': 
				mtl_param_str += 'Pr ' + f.get('value') + '\n'

		# get texture maps and copy the png file to the output directory
		for texture in bsdf_block.findall('texture'):
			# albedo / diffuse
			if texture.get('name') == 'albedo':
				albedo_path = os.path.join(self.brdf_root, texture.find('string').get('value')[33:])
				copyfile(albedo_path, os.path.join(self.save_root, bsdf_block.get('id')+'_diffuse_tiled.png'))
				mtl_param_str += 'map_Kd ' + bsdf_block.get('id') + '_diffuse_tiled.png\n'
			# roughness
			if texture.get('name') == 'roughness':
				roughness_path = os.path.join(self.brdf_root, texture.find('string').get('value')[33:])
				copyfile(roughness_path, os.path.join(self.save_root, bsdf_block.get('id')+'_rough_tiled.png'))
				mtl_param_str += 'map_Pr ' + bsdf_block.get('id') + '_rough_tiled.png\n'
		return mtl_param_str
	
	# the xml tree consists of a set of shape blocks and bsdf blocks
	# each bsdf block corresponds to one material
	# each shape block corresponds to one object, and may inlcude multiple bsdf blocks (materials)
	# the same shape id could appear multiple times in the scene, each for one object instance
	# is_split: split layout
	def parse(self, split_floor=False, is_split=True, is_floor_replaced=True, is_box_replaced=False, borders=[-1., 0.], border_type="y_border", reverse_two_band=False):
		## clear output directory and recreate an empty directory
		if os.path.isdir(self.save_root): 
			rmtree(self.save_root)
		os.mkdir(self.save_root)

		## get scene xml path
		scene_xml_path = os.path.join(self.scene_root, 'xml', self.scene_id, 'main.xml')

		## split layout
		if is_split:
			# create a temporary folder in the save root
			tmp_root = os.path.join(self.save_root, 'tmp')
			os.mkdir(tmp_root)
			# split layout elements
			scene_xml_path = split_layout(self.scene_id, scene_xml_path, os.path.join(self.layoutMesh_root, self.scene_id), tmp_root)
			# duplicate floors
			#materials = ['Material__ceramic_small_diamond', 'Material__roughcast_sprayed', 'Material__ceramic_small_diamond']
			if split_floor:
				#materials = ['Material__carpet_loop', 'Material__sls_alumide_polished_rosy_red', 'Material__carpet_loop']
				if reverse_two_band:
					materials = ['Material__carpet_loop', 'Material__carpet_loop', 'Material__sls_alumide_polished_rosy_red']
				else:	
					materials = ['Material__sls_alumide_polished_rosy_red', 'Material__sls_alumide_polished_rosy_red', 'Material__carpet_loop']
				#borders = [3., 4.]
				borders = borders
				duplicate_floor(scene_xml_path, materials, borders, border_type=border_type)

		## get scene xml file
		self.xml_root = self.__get_scene_xml(scene_xml_path)

		## parse scene xml
		for shape_block in tqdm(self.xml_root.findall('shape'), desc='shape block'):
			obj = self.parse_shape_block(shape_block)
			if obj == None: 
				continue
			self.obj_list.append(obj)

		
		#print("*****************************************")
		#for o in self.obj_list:
		#	print(o.id)	
		#print("*****************************************")
		
		
		## replace floor with planar object
		if is_floor_replaced:
			floor = FloorBBox()
			floor_cnt = 0
			for obj in self.obj_list:
				if obj.id.lower().find('floor') != -1:
					floor_obj_file_name = os.path.join(self.save_root, obj.obj_path)
					floor.parse_obj(floor_obj_file_name)
					floor.generate_floor(floor_obj_file_name, floor_thickness=0.05)
					floor_cnt += 1
		print("%d floors replaced with planar objects!"%(floor_cnt))
		
		## replace box with rectangle object
		if is_box_replaced:
			box = BoxBBox()
			box_cnt = 0
			for obj in self.obj_list:
				if obj.obj_path == '03337140_2f449bf1b7eade5772594f16694be05_object.obj':
					box_obj_file_name = os.path.join(self.save_root, obj.obj_path)
					box.parse_obj(box_obj_file_name)
					box.generate_box(box_obj_file_name)
					box_cnt += 1
		print("%d boxes replaced with rectangle objects!"%(box_cnt))

		print('-------------------------------------')
		print('Parsing Done.')
		print('Scene id: %s, Total: %d objects'%(self.scene_id, len(self.obj_list)))
		print('Output folder: %s'%(self.save_root))
		print('-------------------------------------')

		return self.obj_list
	
	# save object list
	def save_param(self, pickle_path=None):
		if pickle_path == None: 
			pickle_path = os.path.join(self.save_root, 'SceneParser_param.pkl')
		with open(pickle_path, 'wb') as f:
			pickle.dump({
				'scene_id': self.scene_id,
				'save_root': self.save_root,
				'obj_list': self.obj_list,
				'xml_root': self.xml_root
			}, f)
			print(f"[SceneParser.save_param]SceneParser parameters saved to '{pickle_path}'!")
	
	# load object list
	def load_param(self, pickle_path):
		assert os.path.isfile(pickle_path), f"[SceneParser.load_param]Error: '{pickle_path}' is not a file!"
		with open(pickle_path, 'rb') as f:
			param = pickle.load(f)
		
		self.scene_id = param['scene_id']
		self.save_root = param['save_root']
		self.obj_list = param['obj_list']
		self.xml_root = param['xml_root']
		print(f"[SceneParser.load_param]SceneParser parameters loaded from '{pickle_path}'!")

	def print_param(self):
		for idx, obj in enumerate(self.obj_list):
			print(f"========= Obj {idx} =========")
			print(f"obj.id = {obj.id}")
			print(f"obj.mtl_path = {obj.mtl_path}")
			print(f"obj.obj_path = {obj.obj_path}")
			print(f"obj.transfroms = {obj.transforms}")
			# print(f"obj.rotate = {obj.rotate}")
			# print(f"obj.scale = {obj.scale}")
			# print(f"obj.translate = {obj.translate}")
			print(f"obj.type = {obj.type}\n")
		
		print(f"len(obj_list) = {len(self.obj_list)}")    

if __name__ == '__main__':
	### PARSER INPUT PARAMETERS ###
	scene_id = 'scene0420_01'
	kwargs = {
		'scene_root': '../dataset/original/scenes',
		'brdf_root' : '../dataset/original/BRDFOriginDataset',
		'uvMapped_root': '../dataset/original/uv_mapped',
		'envDataset_root': '../dataset/original/EnvDataset',
		'layoutMesh_root': '../dataset/original/layoutMesh'
	}
	# save_root = '../dataset/interactive'
	save_root = '../dataset'
	pickle_path = os.path.join(save_root, scene_id, 'param.pkl')
	
	### PARSE ###
	parser = SceneParser(scene_id, save_root, **kwargs)
	# obj_list = parser.parse()
	parser.parse()
	## save parameters to a pickle file
	parser.save_param(pickle_path)
	
	### EXAMPLE OF OBJ ATTRIBUTES ###
	## load parameters from a pickle file
	parser.load_param(pickle_path)
	parser.print_param()
	
