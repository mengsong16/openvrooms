from openvrooms.data_processor.split import  XmlSplit
from copy import deepcopy as dcp
from gibson2.render.mesh_renderer import tinyobjloader as tiny
import os
import numpy as np

def duplicate_floor(xml_dir, materials: list, borders: list, border_type):
    assert len(materials) == len(borders) + 1, "Number of materials doesn't match number of borders!"
    xmlsplit = XmlSplit(xml_dir)

    ## find floor shape block
    shape_blocks = xmlsplit.find_block('floor', 'shape')
    assert len(shape_blocks) == 1, "More than 1 floor shape blocks!"
    shape_block = shape_blocks[0]
    # block id
    sid = shape_block.get('id')

    ## obj file
    # obj file path
    original_path = shape_block.find('string').get('value')
    # vertices bounds
    reader = tiny.ObjReader()
    ret = reader.ParseFromFile(original_path)
    assert ret, "obj file not read!"
    vertices = np.array(reader.GetAttrib().vertices).reshape((-1, 3)) # (N, 3)
    [x1, y1, z1] = np.max(vertices, axis=0) # (3, )
    [x0, y0, z0] = np.min(vertices, axis=0) # (3, )
    #print(x1,y1,z1)
    #print(x0,y0,z0)
    layout_x0 = -3.418463
    layout_x1 = 3.532937
    layout_y0 = -2.803233
    layout_y1 = 2.722267


    if border_type == "x_border":
        borders[0] = borders[0] - layout_x0 + x0
        borders[1] = borders[1] - layout_x0 + x0

        assert min(borders) >= x0, f"min of borders is less than {x0}!"
        assert max(borders) <= x1, f"max of borders is larger than {x1}!"
        borders = [x0] + sorted(borders) + [x1]
    elif border_type == "y_border":
        borders[0] = borders[0] - layout_y0 + y0
        borders[1] = borders[1] - layout_y0 + y0

        assert min(borders) >= y0, f"min of borders is less than {y0}!"
        assert max(borders) <= y1, f"max of borders is larger than {y1}!"
        borders = [y0] + sorted(borders) + [y1]   
    else:
        print("Error: not known border type")
    

    # obj lines
    with open(original_path, 'r') as f:
        lines = f.readlines()

    ## find floor bsdf block
    bsdf_blocks = xmlsplit.find_block('floor', 'bsdf')
    assert len(bsdf_blocks) == 1, "More than 1 floor bsdf blocks!"
    bsdf_block = bsdf_blocks[0]

    for i in range(len(materials)):
        ## duplicate shape block
        sblock = dcp(shape_block)

        ## change shape block id
        sblock.set('id', sid + str(i+1))

        ## change obj file path
        new_path = os.path.join(os.path.dirname(original_path), sid+str(i+1)+'.obj')
        sblock.find('string').set('value', new_path)

        ## duplicate obj file & clip obj vertices
        vmin, vmax = borders[i], borders[i+1]
        new_lines = list()
        for line in lines:
            if line.startswith('v '):
                vertex = np.array([float(value) for value in line.strip().split()[1:]], dtype=float)
                if border_type == "x_border":
                    vertex[0] = np.clip(vertex[0], vmin, vmax)
                else:
                    vertex[1] = np.clip(vertex[1], vmin, vmax)   
                new_lines.append('v ' + ' '.join(str(e) for e in vertex) + '\n')
            elif line.startswith('usemtl'):
                new_lines.append('usemtl ' + sid + str(i+1) + '\n')
            else:
                new_lines.append(line)
        with open(new_path, 'w') as f:
            f.write(''.join(new_lines))

        ## change bsdf block id in shape block
        sblock.find('ref').set('id', sid + str(i+1))
        
        ## add shape block to the tree
        xmlsplit.root.append(sblock)

        ## duplicate bsdf block
        bblock = dcp(bsdf_block)

        ## change bsdf block id
        bblock.set('id', sid + str(i+1))

        ## change material
        material_name = materials[i]
        for s in bblock.iter('string'):
            img_dir = s.get('value')
            start_idx = img_dir.find('/', img_dir.find('BRDFOriginDataset'))
            end_idx = img_dir.find('/', start_idx+1)
            assert start_idx != -1 and end_idx != -1, "material folder not found!"
            s.set('value', img_dir[:start_idx+1] + material_name + img_dir[end_idx:])
        
        ## reset albedoScale
        for rgb in bblock.iter('rgb'):
            if rgb.get('name') == 'albedoScale':
                rgb.set('value', '1.0 1.0 1.0')
        
        ## reset uvScale
        #for flt in bblock.iter('float'):
        #    if flt.get('name') == 'uvScale':
        #        flt.set('value', '1.0')
        
        ## add bsdf block to the tree
        xmlsplit.root.append(bblock)
        
    ## delete original shape block
    xmlsplit.root.remove(shape_block)

    ## delete original bsdf block
    xmlsplit.root.remove(bsdf_block)

    ## delete original obj file  
    os.remove(original_path)

    ## save xml
    xmlsplit.save_xml(xml_dir)  

if __name__ == '__main__':
    xml_dir = '../tmp/main.xml'
    materials = ['Material__ceramic_small_diamond', 'Material__roughcast_sprayed', 'Material__ceramic_small_diamond']
    borders = [3., 4.]
    duplicate_floor(xml_dir, materials, borders)
