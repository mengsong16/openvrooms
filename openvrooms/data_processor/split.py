import xml.etree.ElementTree as ET
from shutil import rmtree
from copy import deepcopy as dcp
from shutil import copyfile, rmtree
import os

class XmlSplit:
    def __init__(self, xml_dir: str):
        self.tree = ET.parse(xml_dir)
        self.root = self.tree.getroot()
    def find_block(self, target_block_id: str, target_block_class='shape') -> list:
        blocks = list()
        for block in self.root.findall(target_block_class):
            block_id = block.get('id')
            if block_id.lower().find(target_block_id) != -1:
                blocks.append(block)
        return blocks
    def split_shape_block(self, target_shape_block, objfile_save_path):
        # construct a shape block for each component
        component_ids = list()
        for ref in target_shape_block.findall('ref'):
            # get component id
            component_id = ref.get('id')
            component_ids.append(component_id)
            # copy shape block
            component_block = dcp(target_shape_block)
            # change block id
            component_block.set('id', component_id)
            # change obj file path
            component_block.find('string').set('value', os.path.join(objfile_save_path, component_id+'.obj'))
            # remove other components
            for cref in component_block.findall('ref'):
                if cref.get('id') != component_id:
                    component_block.remove(cref)
            # add the component block to the tree
            self.root.append(component_block)
        # remove the original layout block
        self.root.remove(target_shape_block)
    def save_xml(self, xml_save_dir: str):
        # write the new xml file
        self.tree.write(xml_save_dir)

class ObjSplit:
    def __init__(self, objfile_path: str):
        self.__parse_objfile(objfile_path)
    def __parse_objfile(self, objfile_path: str):
        with open(objfile_path, 'r') as f:
            lines = f.readlines()
        vs, vts, vns = [None], [None], [None] # [(line index, type index)]
        mtl_map = dict()
        for lidx, line in enumerate(lines):
            if line.startswith('v '):
                vs.append((lidx, len(vs)))
            if line.startswith('vt'):
                vts.append((lidx, len(vts)))
            if line.startswith('vn'):
                vns.append((lidx, len(vns)))
            if line.startswith('usemtl'):
                mtl = line.strip()[7:]
                assert mtl not in mtl_map, f"[ObjSplit]Error: repeated material: {mtl}!"
                mtl_map[mtl] = {'vidxs': set(), 'vtidxs': set(), 'vnidxs': set(), 'lidxs': [lidx]}
            if line.startswith('f '):
                if len(mtl_map) == 0: continue
                mtl_map[mtl]['lidxs'].append(lidx)
                face = line.strip().split()[1:] # ['v1/vt1/vn1', 'v2/vt2/vn2', ...]
                for idx_str in face:
                   vidx, vtidx, vnidx = map(int, idx_str.split('/'))
                   mtl_map[mtl]['vidxs'].add(vidx)
                   mtl_map[mtl]['vtidxs'].add(vtidx)
                   mtl_map[mtl]['vnidxs'].add(vnidx)
        self.lines = lines
        self.vs, self.vts, self.vns = vs, vts, vns
        self.mtl_map = mtl_map
    r'''
        construct for a material's attribute x \in {v, vt, vn}:
        [line index] for a material's attribute x
        {old_idx: new_idx (after deleting other materials' attributes)}

        xs: [(line index, old type index)] for all materials attribute x
        xidxs: {old type index} for a material's attribute x
    ''' 
    def find_mtl_attrib(self, xs: list, xidxs: set):
        mtl_attrib = [xs[xidx] for xidx in sorted(xidxs)] # [(line index, old type index)] for a material's attribute x
        mtl_attrib_lidxs = list()
        old2new_idx_map = dict()
        for new_xidx, attrib in enumerate(mtl_attrib):
            lidx, old_xidx = attrib
            mtl_attrib_lidxs.append(lidx)
            old2new_idx_map[old_xidx] = new_xidx+1
        return mtl_attrib_lidxs, old2new_idx_map
    def extract_mtl(self, mtl: str):
        assert mtl in self.mtl_map.keys(), f"[ObjSplit.extract_mtl]Error: material {mtl} doesn't exist!"
        vidxs = self.mtl_map[mtl]['vidxs']   # set of old v attribute indices corresponding to mtl
        vtidxs = self.mtl_map[mtl]['vtidxs'] # set of old vt attribute indices corresponding to mtl
        vnidxs = self.mtl_map[mtl]['vnidxs'] # set of old vn attribute indices corresponding to mtl
        lidxs = self.mtl_map[mtl]['lidxs']   # list of line indices of the f block corresponding to mtl
        v_lidxs, v_old2new = self.find_mtl_attrib(self.vs, vidxs)     # list of line indices of v attribute corresponding to mtl, dict of old to new v attribute indices
        vt_lidxs, vt_old2new = self.find_mtl_attrib(self.vts, vtidxs)
        vn_lidxs, vn_old2new = self.find_mtl_attrib(self.vns, vnidxs)
        # edited lines of the original object file containing only entries corresponding to mtl
        lines = self.lines.copy()
        # mark attribute lines corresponding to other mtl's as to be deleted
        for lidx, line in enumerate(lines):
            if line.startswith('v ') and lidx not in v_lidxs:
                lines[lidx] = None
            elif line.startswith('vt') and lidx not in vt_lidxs:
                lines[lidx] = None
            elif line.startswith('vn') and lidx not in vn_lidxs:
                lines[lidx] = None
        # mark f blocks corresponding to other mtl's as to be deleted
        for mtl_, mtl_attrib_ in self.mtl_map.items():
            if mtl_ == mtl: continue
            lidxs_ = mtl_attrib_['lidxs']
            for lidx_ in lidxs_:
                lines[lidx_] = None
        # change attribute indices in f block
        for lidx in lidxs[1:]:
            line = lines[lidx]
            face = line.strip().split()[1:] # ['v1/vt1/vn1', 'v2/vt2/vn2', ...]
            new_line = 'f'
            for idx_str in face:
                vidx, vtidx, vnidx = map(int, idx_str.split('/'))
                new_line += ' ' + '/'.join([str(v_old2new[vidx]), str(vt_old2new[vtidx]), str(vn_old2new[vnidx])])
            lines[lidx] = new_line + '\n'
        lines = list(filter(None.__ne__, lines))
        return lines
    def split_mtls(self, save_dir: str):
        assert os.path.isdir(save_dir), f"[ObjSplit.split_mtls]Error: save directory {save_dir} doesn't exist!"
        for mtl in self.mtl_map.keys():
            # extract lines from the original object file
            mtl_lines = self.extract_mtl(mtl)
            with open(os.path.join(save_dir, mtl+'.obj'), 'w') as f:
                f.write(''.join(mtl_lines))

def split_layout(scene_id: str, in_xml_dir: str, in_objfile_path: str, save_path: str):
    ## names
    in_xml_name = in_xml_dir.split('/')[-1]

    ## paths
    xml_save_dir = os.path.join(save_path, 'main.xml')

    ## split xml layout block
    xml_split = XmlSplit(in_xml_dir)
    shape_blocks = xml_split.find_block(target_block_id=scene_id, target_block_class='shape')
    for shape_block in shape_blocks:
        objfile_dir = shape_block.find('string').get('value')
        if objfile_dir.lower().find('container') == -1:
            break
    xml_split.split_shape_block(shape_block, save_path)
    xml_split.save_xml(xml_save_dir)

    ## split obj materials
    objfile_name = objfile_dir.split('/')[-1]
    obj_split = ObjSplit(os.path.join(in_objfile_path, objfile_name))
    obj_split.split_mtls(save_path)

    return xml_save_dir

if __name__ == '__main__':
    ## hyper parameters
    scene_id = 'scene0420_01'
    in_xml_dir = './main.xml'
    in_objfile_path = './'
    save_path = 'scene0420_01/tmp'

    ## split layout
    print(split_layout(scene_id, in_xml_dir, in_objfile_path, save_path))
