import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# axis_angle     : [angle, x, y, z]
# rotation vector: rad(angle) * [x, y, z]
def axisAngle2rotVec(axis_angle: np.ndarray) -> np.ndarray:
    assert axis_angle.shape == (4, ), f"ndarray 'axis_angle' has invalid shape: {axis_angle.shape}!"
    angle = axis_angle[0] # scalar
    axis = axis_angle[1:] # (3, )
    rad = angle * np.pi / 180
    return rad * axis
# axis_angle: [angle, x, y, z]
# quaternion: [vec(3), s] s.t. q = vec[0]i + vec[1]j + vec[2]k + s
def axisAngle2quat(axis_angle: np.ndarray) -> np.ndarray:
    rotate = R.from_rotvec(axisAngle2rotVec(axis_angle))
    return rotate.as_quat()

class ObjTransformBasic:
    def __init__(self, filename: str):
        self.lines = list()
        self.vertices, self.vertex_line_indices = list(), list()
        self.normals, self.normal_line_indices = list(), list()
        self.vt2mtl = dict() # map indices of vt's (int) to the corresponding material names (str)
        self.mtl2vt = dict() # map material names (str) to list of vt indices (int)
        self.tex_coords = list()
        self.__parse_obj_file(filename)
    def __parse_obj_file(self, filename: str):
        with open(filename, 'r') as f:
            lines = f.readlines()
        vertex_list, vertex_line_indices = list(), list()
        normal_list, normal_line_indices = list(), list()
        tex_coord_list= list()
        vt2mtl = dict()
        mtl2vt = dict()
        for idx, line in enumerate(lines):
            if line.startswith('v '):
                vertex = np.array([float(value) for value in line.strip().split()[1:]], dtype=float)
                assert vertex.shape == (3, ) or vertex.shape == (6, ), f"vertex {vertex} has invalid shape!"
                vertex_list.append(vertex[:3])
                vertex_line_indices.append(idx)
            if line.startswith('vn'):
                normal = np.array([float(value) for value in line.strip().split()[1:]], dtype=float)
                assert normal.shape == (3, ), f"normal {normal} has invalid shape!"
                normal_list.append(normal)
                normal_line_indices.append(idx)
            if line.startswith('vt'):
                tex_coord = np.array([float(value) for value in line.strip().split()[1:]], dtype=float)
                assert tex_coord.shape == (2, ), f"texture coordinate {tex_coord} has invalid shape!"
                tex_coord_list.append(tex_coord)
            if line.startswith('usemtl'):
                line = line.strip()
                current_mtl = line[7:]
            if line.startswith('f '):
                face = line.strip().split()
                for idx_triple in face[1:]:
                    vt_idx = int(idx_triple.split('/')[1])
                    vt2mtl[vt_idx-1] = current_mtl
                    if not current_mtl in mtl2vt: mtl2vt[current_mtl] = list()
                    mtl2vt[current_mtl].append(vt_idx-1)
        assert len(vertex_list) == len(vertex_line_indices), f"the number of vertices {len(vertex_list)} doesn't match the number of indices {len(vertex_line_indices)}!"
        assert len(normal_list) == len(normal_line_indices), f"the number of normals {len(normal_list)} doesn't match the number of indices {len(normal_line_indices)}!"
        vertices = np.array(vertex_list, dtype=float)
        normals = np.array(normal_list, dtype=float)
        tex_coords = np.array(tex_coord_list, dtype=float)
        self.lines, self.vertices, self.vertex_line_indices, self.normals, self.normal_line_indices = lines, vertices, vertex_line_indices, normals, normal_line_indices
        self.vt2mtl = vt2mtl
        self.mtl2vt = mtl2vt
        self.tex_coords = tex_coords
    def print_parsed_elements(self):
        print(f"=== vertices [{self.vertices.shape}] ===")
        for i in range(self.vertices.shape[0]):
            print("v " + " ".join(self.vertices[i, :].astype(str)))
        print(f"=== normals [{self.normals.shape}] ===")
        for i in range(self.normals.shape[0]):
            print("vn " + " ".join(self.normals[i, :].astype(str)))
    def form_obj_file(self, overwrite=True):
        lines = self.lines.copy()
        # vertices (v)
        assert self.vertices.ndim == 2 and self.vertices.shape[1] == 3, f"ndarray 'vertices' has invalid shape: {self.vertices.shape}!"
        for i in range(self.vertices.shape[0]):
            lines[self.vertex_line_indices[i]] = "v " + " ".join(self.vertices[i, :].astype(str)) + "\n"
        # normals (vn)
        assert self.normals.ndim == 2 and self.normals.shape[1] == 3, f"ndarray 'normals' has invalid shape: {self.normals.shape}!"
        for i in range(self.normals.shape[0]):
            lines[self.normal_line_indices[i]] = "vn " + " ".join(self.normals[i, :].astype(str)) + "\n"
        if overwrite: self.lines = lines
        return lines
    def save_obj_file(self, save_path: str):
        with open(save_path, 'w') as f:
            f.writelines(self.lines)
    def remove_lines(self, startswith: str, overwrite=True):
        lines = self.lines.copy()
        for idx, line in enumerate(lines):
            if line.startswith(startswith):
                lines[idx] = None
        # filter out None
        lines = list(filter(None.__ne__, lines))
        if overwrite: self.lines = lines
        return lines
    def duplicate_faces(self, overwrite=True):
        lines = self.lines.copy()
        new_lines = list()
        for line in lines:
            if line.startswith('f '):
                new_lines.append(line)
                reversed_face = line.strip().split()[1:]
                reversed_face.reverse()
                new_lines.append(' '.join(['f']+reversed_face)+'\n')
            else:
                new_lines.append(line)
        if overwrite: self.lines = new_lines
        return new_lines
    def transform_vertices_and_normals(self, transform_list: list, overwrite=True):
        vertices = np.array(self.vertices)
        normals = np.array(self.normals)
        num_vertices = vertices.shape[0]
        num_normals = normals.shape[0]
        assert vertices.shape == (num_vertices, 3), f"ndarray 'vertices' has invalid shape: {vertices.shape}!"
        assert normals.shape == (num_normals, 3), f"ndarray 'normals' has invalid shape: {normals.shape}!"
        for transform in transform_list:
            if transform[0].lower() == 'scale':
                scale = transform[1]
                assert scale.shape == (3, ), f"'scale' transformation has invalid shape: {scale.shape}!"
                vertices = vertices * scale.reshape((1, 3))
                normals = normals * scale.reshape((1, 3))
            elif transform[0].lower() == 'translate':
                translate = transform[1]
                assert translate.shape == (3, ), f"'translate' transformation has invalid shape: {translate.shape}!"
                vertices += translate.reshape((1, 3))
            elif transform[0].lower() == 'rotate':
                if transform[1].lower() == 'axisangle':
                    axis_angle = transform[2]
                    rotate = R.from_rotvec(axisAngle2rotVec(axis_angle))
                elif transform[1].lower() == 'quaternion':
                    quaternion = transform[2]
                    assert quaternion.shape == (4, ), f"ndarray 'quaternion' has invalid shape {quaternion.shape}!"
                    rotate = R.from_quat(quaternion)
                elif transform[1].lower() == 'rotationmatix':
                    rotation_matrix = transform[2]
                    assert rotation_matrix.shape == (3, 3), f"ndarray 'rotation matrix' has invalid shape {rotation_matrix.shape}!"
                    rotate = R.from_matrix(rotation_matrix)
                else:
                    raise NotImplementedError(f"Not implemented rotation representation {transform[1]}!")
                vertices = rotate.apply(vertices)
                normals = rotate.apply(normals)
            else:
                raise NotImplementedError(f"Not implemented transformation {transform[0]}!")
        # normalize the 'normals'
        normals = normals / np.linalg.norm(normals, axis=1).reshape((-1, 1))
        # round values
        vertices = np.around(vertices, decimals=6)
        normals = np.around(normals, decimals=6)
        assert vertices.shape == (num_vertices, 3), f"transformed 'vertices' has inconsistent shape: {vertices.shape}!"
        assert normals.shape == (num_normals, 3), f"transformed 'normals' has inconsistenct shape: {normals.shape}!"
        if overwrite:
            self.vertices = vertices
            self.normals = normals
        return vertices, normals
    def apply_uvScale(self, mtl: str, uvScale: float, overwrite=True):
        tex_coords = np.array(self.tex_coords)
        if mtl in self.mtl2vt:
            vt_indices = self.mtl2vt[mtl]
            mtl_vts = tex_coords[vt_indices, :] * uvScale
            tex_coords[vt_indices, :] = mtl_vts
        if overwrite: self.tex_coords = tex_coords
        return tex_coords
    def calc_tex_coord_range(self):
        mtl2range = dict()
        for mtl, vt_indices in self.mtl2vt.items():
            mtl_vts = self.tex_coords[vt_indices, :]
            (umin, vmin), (umax, vmax) = mtl_vts.min(axis=0), mtl_vts.max(axis=0)
            mtl2range[mtl] = ((umin, vmin), (umax, vmax))
        return mtl2range
    def map_vt_to_combined_texture(self, mtl_list: list, texture_size: int, mtl2uvScale: dict, overwrite=True):
        mtl2range = self.calc_tex_coord_range()
        new_lines = list()
        mtl_set = False 
        vt_idx = -1
        for line in self.lines:
            if line.startswith('usemtl'): 
                if not mtl_set:
                    new_lines.append('usemtl combined_material\n')
                    mtl_set = True
            elif line.startswith('vt'):
                vt_idx += 1
                _, u, v = line.strip().split()
                u, v = float(u), float(v)
                if not self.vt2mtl[vt_idx] in mtl_list: 
                    new_lines.append(f'vt {u} {v}\n')
                    continue
                mtl = self.vt2mtl[vt_idx]
                mtl_idx = len(mtl_list) - mtl_list.index(mtl) - 1 # mtl_list.index(mtl)
                # calculate the range of texture coordinates
                ((umin, vmin), (umax, vmax)) = mtl2range[mtl]
                # normalize u, v values
                uvScale = 1.
                if mtl in mtl2uvScale: uvScale = mtl2uvScale[mtl]
                low, high = (mtl_idx) / len(mtl_list), (mtl_idx+1) / len(mtl_list)
                # low, high = (mtl_idx-2) / len(mtl_list), (mtl_idx-1) / len(mtl_list)
                new_u = u * uvScale # (u - umin) / (umax - umin)
                new_v = (v * uvScale - vmin) * (high - low) / (vmax - vmin) + low
                # tqdm.write(f"ObjTrans: {mtl}|({low}, {high})|{u}=>({umin}, {umax})=>{new_u}|{v}=>({vmin}, {vmax})=>{new_v}")
                new_lines.append(f'vt {new_u} {new_v}\n')
            elif line.startswith('g '):
                pass
            else:
                new_lines.append(line)
        if overwrite: self.lines = new_lines
        return new_lines

class ObjTransform(ObjTransformBasic):
    def __init__(self, filename: str):
        self.mtllib = list()
        self.tex_coords = list()
        self.f_blocks = list()
        self.__parse_obj_file(filename)
    def __parse_obj_file(self, filename: str):
        with open(filename, 'r') as f:
            lines = f.readlines()
        mtllib_list = list()
        vertex_list, vertex_line_indices = list(), list()
        normal_list, normal_line_indices = list(), list()
        tex_coord_list = list()
        f_blocks = list()
        f_block = list()

        for idx, line in enumerate(lines):
            if line.startswith('mtllib'): mtllib_list.append(line)
            if line.startswith('v '): 
                vertex = np.array([float(value) for value in line.strip().split()[1:]], dtype=float)
                assert vertex.shape == (3, ) or vertex.shape == (6, ), f"vertex {vertex} has invalid shape!"
                vertex_list.append(vertex[:3])
                vertex_line_indices.append(idx)
            if line.startswith('vn'):
                normal = np.array([float(value) for value in line.strip().split()[1:]], dtype=float)
                assert normal.shape == (3, ), f"normal {normal} has invalid shape!"
                normal_list.append(normal)
                normal_line_indices.append(idx)
            if line.startswith('vt'): tex_coord_list.append(line)
            if line.startswith('f '): 
                assert len(f_block) > 0, f"'usemtl' line missing while appending an 'f' line [{idx}]!"
                f_block.append(line)
            if line.startswith('usemtl'):
                if len(f_block) == 0: f_block.append(line)
                elif len(f_block) == 1:
                    if f_block[0].startswith('usemtl'): print(f"consecutive 'usemtl' line without an 'f' line [{idx}] in between!")
                    else: print(f"invalid starting entry in an 'f' block: {f_block[0]}!")
                    exit()
                else:
                    assert f_block[0].startswith('usemtl'), f"invalid starting entry in an 'f' block: {f_block[0]}!"
                    f_blocks.append(f_block)
                    f_block = [line]
        assert len(f_block) > 1, f"final 'f' block has invalid number of entries: {len(f_block)}!"
        f_blocks.append(f_block)
        
        vertices = np.array(vertex_list, dtype=float)
        normals = np.array(normal_list, dtype=float)
        self.lines, self.vertices, self.vertex_line_indices, self.normals, self.normal_line_indices = lines, vertices, vertex_line_indices, normals, normal_line_indices
        self.mtllib, self.tex_coords, self.f_blocks = mtllib_list, tex_coord_list, f_blocks
    def print_parsed_elements(self):
        print(f"=== mtllib [{len(self.mtllib)}] ===")
        print("".join(self.mtllib))
        print(f"=== vertices [{self.vertices.shape}] ===")
        for i in range(self.vertices.shape[0]):
            print("v " + " ".join(self.vertices[i, :].astype(str)))
        print(f"=== normals [{self.normals.shape}] ===")
        for i in range(self.normals.shape[0]):
            print("vn " + " ".join(self.normals[i, :].astype(str)))
        print(f"=== tex_coords [{len(self.tex_coords)}] ===")
        print("".join(self.tex_coords))
        print(f"=== f_blocks [{len(self.f_blocks)}: {[len(f_block)-1 for f_block in self.f_blocks]}] ===")
        print("".join(["".join(f_block) for f_block in self.f_blocks]))
    def save_obj_file(self, save_path: str):
        with open(save_path, 'w') as f:
            # mtllib
            f.write("".join(self.mtllib))
            # vertices (v)
            assert self.vertices.ndim == 2 and self.vertices.shape[1] == 3, f"ndarray 'vertices' has invalid shape: {self.vertices.shape}!"
            for i in range(self.vertices.shape[0]):
                f.write("v " + " ".join(self.vertices[i, :].astype(str)) + '\n')
            # normals (vn)
            assert self.normals.ndim == 2 and self.normals.shape[1] == 3, f"ndarray 'normals' has invalid shape: {self.normals.shape}!"
            for i in range(self.normals.shape[0]):
                f.write("vn " + " ".join(self.normals[i, :].astype(str)) + '\n')
            # texture coordinates (vt)
            f.write("".join(self.tex_coords))
            # f blocks 
            f.write("".join(["".join(f_block) for f_block in self.f_blocks]))


if __name__ == '__main__':
    filename = '../dataset/interactive-original/scene0420_01/ceiling_lamp_15650_object_aligned_light.obj'
    objTrans = ObjTransformBasic(filename)
    objTrans.transform_vertices_and_normals([('translate', np.array([100, 0, 0])), ('rotate', 'axisangle', np.array([180, 1., 0, 0])), ('scale', np.array([0.5, 1, 1]))])
    objTrans.save_obj_file('parsed.obj')


    
