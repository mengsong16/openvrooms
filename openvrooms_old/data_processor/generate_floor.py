import os
import xml.etree.ElementTree as ET
from shutil import rmtree

from openvrooms.config import *
from openvrooms.data_processor.xml_parser import SceneParser
from openvrooms.data_processor.split import XmlSplit, split_layout
from openvrooms.data_processor.floor_bbox import FloorBBox

class FloorParser(SceneParser):
    def parse(self, material_name=None, is_floor_replaced=True):
        ## clear output directory and recreate an empty directory
        if os.path.isdir(self.save_root): 
            rmtree(self.save_root)
        os.mkdir(self.save_root)

        ## get scene xml path
        scene_xml_path = os.path.join(self.scene_root, 'xml', self.scene_id, 'main.xml')

        ## create a temporary folder in the save root
        tmp_root = os.path.join(self.save_root, 'tmp')
        os.mkdir(tmp_root)
        # split layout elements
        scene_xml_path = split_layout(self.scene_id, scene_xml_path, os.path.join(self.layoutMesh_root, self.scene_id), tmp_root)

        ## change material image path in the new xml (i.e. the xml with split layouts)
        if material_name != None:
            # find the floor bsdf block
            xml_split = XmlSplit(scene_xml_path)
            floor_bsdf_blocks = xml_split.find_block(target_block_id='floor', target_block_class='bsdf')
            assert len(floor_bsdf_blocks) == 1, f"[FloorParser.parse]Error: multiple ({len(floor_bsdf_blocks)}) floor bsdf blocks found!"
            floor_bsdf = floor_bsdf_blocks[0]
            # modify the material image path
            for s in floor_bsdf.iter('string'):
                img_dir = s.get('value')
                start_idx = img_dir.find('/', img_dir.find('BRDFOriginDataset'))
                end_idx = img_dir.find('/', start_idx+1)
                assert start_idx != -1 and end_idx != -1, "[FloorParser.parse]Error: material fold not found!"
                s.set('value', img_dir[:start_idx+1] + material_name + img_dir[end_idx:])
            xml_split.save_xml(scene_xml_path)

        ## get scene xml file
        self.xml_root = self._SceneParser__get_scene_xml(scene_xml_path)

        ## parse floor shape block
        for shape_block in self.xml_root.findall('shape'):
            if shape_block.get('id').lower().find('floor') != -1:
                obj = self.parse_shape_block(shape_block)
                if obj == None: continue
                self.obj_list.append(obj)
        
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

        ## remove temporary folder
        rmtree(tmp_root)
        
        print('-------------------------------------')
        print('Parsing Done.')
        print('Scene id: %s, Total: %d objects'%(self.scene_id, len(self.obj_list)))
        print('Output folder: %s'%(self.save_root))
        print('-------------------------------------')


def gen_floor(scene_id, material_name):
    # paths
    original_dataset_path = os.path.join(dataset_path, 'original')
    kwargs = {
	    'scene_root': os.path.join(original_dataset_path, 'scenes'),
	    'brdf_root' : os.path.join(original_dataset_path, 'BRDFOriginDataset'),
	    'uvMapped_root': os.path.join(original_dataset_path, 'uv_mapped'),
	    'envDataset_root': os.path.join(original_dataset_path, 'EnvDataset'),
	    'layoutMesh_root': os.path.join(original_dataset_path, 'layoutMesh')
	}
    parser = FloorParser(scene_id=scene_id, save_root=os.path.join(root_path, 'dataset/floor'), **kwargs)
    parser.parse(material_name=material_name, is_floor_replaced=True)

if __name__ == '__main__':
    gen_floor('scene0420_01', 'Material__bricks_royal_range_combined')