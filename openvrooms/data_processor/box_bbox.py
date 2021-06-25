import os
from pathlib import Path
import logging
import numpy as np
from gibson2.render.mesh_renderer import tinyobjloader as tiny
from openvrooms.config import *

class BoxBBox(object):
    def __init__(self, small_box):
        self.reader = None
        self.mtllib = ''
        # predefine v/vt indices for each of the 12 trimesh faces
        self.v_faces = np.array([
                 (1,2,3),
                 (2,4,3),
				 (1,3,5),
				 (3,7,5),
				 (1,5,2),
				 (5,6,2),
				 (2,6,4),
				 (6,8,4),
				 (4,8,7),
				 (7,3,4),
				 (6,7,8),
				 (6,5,7)
                ])

        self.vt_faces = np.array([
                 (1,2,3),
                 (2,4,3),
				 (1,3,2),
				 (3,4,2),
				 (1,3,2),
				 (3,4,2),
				 (4,2,3),
				 (2,1,3),
				 (3,1,2),
				 (2,4,3),
				 (2,3,4),
				 (2,1,3)
                ])

        # small or big
        self.small_box = small_box

    def parse_obj(self, obj_file_name):
        # parse .obj file
        self.reader = tiny.ObjReader()
        ret = self.reader.ParseFromFile(obj_file_name)

        if not ret:
            logging.error("Warning: {}".format(self.reader.Warning()))
            logging.error("Error: {}".format(self.reader.Error()))
            logging.error("Failed to load: {}".format(obj_file_name))
            return

        logging.info("File {} loaded!".format(obj_file_name))
        # get mtllib file
        with open(obj_file_name, 'r') as f:
            lines = f.readlines()
        has_mtllib = False

        for line in lines:
            if line.lower().startswith('mtllib'):
                self.mtllib = line
                has_mtllib = True
                break

        if not has_mtllib:
            logging.error("Warning: no mtllib file found!")
        # check number of materials
        num_mtl = len(self.reader.GetMaterials())
        if num_mtl > 1:
            logging.error("Warning: multiple materials found: {}!".format(num_mtl))
        
        print("Material number: %d"%num_mtl)    
   
    def generate_v_bbox(self):
        if self.reader == None:
            logging.error("Error: no .obj file!")
            return np.array([])

        vertices = np.array(self.reader.GetAttrib().vertices).reshape((-1, 3)) # (N, 3)
        [x1, y1, z1] = np.max(vertices, axis=0) # (3, )
        #y1 = y1 - 0.32
        [x0, y0, z0] = np.min(vertices, axis=0) # (3, )

        if self.small_box:
            y1 = x1 - x0 + y0

        bounds = np.array([
            [x0, y0], 
            [x1, y0],
            [x0, y1],
            [x1, y1]
        ])
        #(z_min, z_max) = (z_max, z_min) if z_max < z_min else (z_min, z_max)
        v_bbox = np.vstack([
            np.hstack([bounds, np.ones((4, 1))*z1]),
            np.hstack([bounds, np.ones((4, 1))*z0])
        ]) # (8, 3)

        return v_bbox
    
    def generate_vt_bbox(self):
        if self.reader == None:
            logging.error("Error: no .obj file!")
            return np.array([])

        coords = np.array(self.reader.GetAttrib().texcoords).reshape((-1, 2)) # (N, 2)
        [u1, v1] = np.max(coords, axis=0) # (2, )
        [u0, v0] = np.min(coords, axis=0) # (2, )
        vt_bbox = np.array([
            [u0, v0], 
            [u0, v1],
            [u1, v0],
            [u1, v1]
        ]) # (4, 2)
        return vt_bbox
    
    def generate_box(self, save_path):
        parent_dir = Path(save_path).parent
        assert os.path.isdir(parent_dir), f"File {parent_dir} doesn't exist!"
        
        v_bbox = self.generate_v_bbox()
        vt_bbox = self.generate_vt_bbox()

        with open(save_path, 'w') as f:
            # write mtllib file
            f.write(self.mtllib)
            # write vertices
            for (x, y, z) in v_bbox:
                f.write('v {} {} {}\n'.format(x, y, z))
            # write vertex coordinates (for material)
            for (u, v) in vt_bbox:
                f.write('vt {} {}\n'.format(u, v))
            # write component material
            if len(self.reader.GetMaterials()) > 0:
                f.write('usemtl {}\n'.format(self.reader.GetMaterials()[0].name))
            # write faces
            for ((v1, v2, v3), (vt1, vt2, vt3)) in zip(self.v_faces, self.vt_faces):
                f.write('f {}/{} {}/{} {}/{}\n'.format(v1, vt1, v2, vt2, v3, vt3))

    '''
    # ---------------- urdf related -----------------------
    def generate_bbox_box_obj(layout_mesh, scene_path):   
        # bounds - axis aligned bounds of mesh
        # 2*3 matrix, min, max, x, y, z
        layout_bounds = layout_mesh.bounds

        floor_thickness = 0.2
        #print(layout_bounds)
        
        floor_bbox = ComponentBBox(layout_bounds[0][0], layout_bounds[1][0], layout_bounds[0][1], layout_bounds[1][1], -floor_thickness, 0.0) 
        #print(floor_bbox.x)
        #print(floor_bbox.y)
        #print(floor_bbox.z)

        # generate floor .obj file
        floor_obj_path = os.path.join(scene_path, "floor.obj")
        floor_bbox.gen_cube_obj(file_path=floor_obj_path, is_color=False, should_save=True)
        print('Generated floor obj file')

        # copy .obj to vhach.obj
        floor_vhacd_obj_path = os.path.join(scene_path, "floor_vhacd.obj")
        shutil.copyfile(floor_obj_path, floor_vhacd_obj_path)   
        print('Generated floor vhacd obj file')
    
    def generate_box_urdf(scene_path):    
        urdf_prototype_file = os.path.join(metadata_path, 'urdf_prototype.urdf') # urdf template
        log_file = os.path.join(scene_path, "vhacd_log.txt")

        builder = ObjectUrdfBuilder(scene_path, log_file=log_file, urdf_prototype=urdf_prototype_file)
        floor_obj_path = os.path.join(scene_path, "floor.obj")
        builder.build_urdf(filename=floor_obj_path, force_overwrite=True, decompose_concave=True, force_decompose=False, mass=100, center=None)  #'geometric'
        print('Generated floor urdf file')

    def floor_collision_detection(robot_id, floor_id):
        collision_links = list(p.getContactPoints(bodyA=robot_id, bodyB=floor_id))
        for item in collision_links:
            print('bodyA:{}, bodyB:{}, linkA:{}, linkB:{}'.format(item[1], item[2], item[3], item[4]))
        
        return len(collision_links) > 0 


    def load_box(scene_path):
        floor_urdf_file = os.path.join(scene_path, "box.urdf")
        floor_id = p.loadURDF(fileName=floor_urdf_file, useFixedBase=1)

        #p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        #floor_id = p.loadURDF("plane.urdf")

        # change floor color
        p.changeVisualShape(objectUniqueId=floor_id, linkIndex=-1, rgbaColor=[0.86,0.86,0.86,1])

        floor_pos, floor_orn = p.getBasePositionAndOrientation(floor_id)

        print("Floor position: %s"%str(floor_pos))
        print("Floor orientation: %s"%str(floor_orn))

        return floor_id

    def test_floor_urdf(scene_path):
        time_step = 1./240. 
        p.connect(p.GUI) # load with pybullet GUI
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(time_step)

        # load floor or scene
        floor_id = load_floor(scene_path)
        
        #scene = NavigateScene(scene_id='scene0420_01', n_obstacles=0)
        #scene.load()
        #floor_id = scene.floor_id
        

        # load robot
        robot_config = parse_config(os.path.join(config_path, "turtlebot_interactive_demo.yaml"))
        turtlebot = Turtlebot(config=robot_config, robot_urdf=turtlebot_urdf_file) 

        robot_ids = turtlebot.load()
        robot_id = robot_ids[0]

        turtlebot.set_position([0, 0, 0])
        turtlebot.robot_specific_reset()
        turtlebot.keep_still() 

        collision_counter = 0
        # start simulation
        
        # keep still
        for _ in range(100):
            p.stepSimulation()
            #collision_counter += floor_collision_detection(robot_id, floor_id)
            time.sleep(time_step) # this is just for visualization, could be removed without affecting avoiding initial collisions
        
        
        # move    
        time_step_n = 0
        for _ in range(50):  # at least 100 seconds
            action = np.random.uniform(-1, 1, turtlebot.action_dim)
            turtlebot.apply_action(action)
            p.stepSimulation()
            time_step_n += 1
            print('----------------------------------------')
            print('time step: %d'%(time_step_n))
            collision_counter += floor_collision_detection(robot_id, floor_id)
            time.sleep(time_step)

        print("Collision steps:%d"%(collision_counter))
        
        p.disconnect()
    '''    

if __name__ == '__main__':
    #obj_file_name = 'scene0420_01_box.obj'
    #save_path = 'box.obj'
    obj_file_name = os.path.join(original_dataset_path, "uv_mapped/03337140/2f449bf1b7eade5772594f16694be05/alignedNew.obj")
    save_path = "/home/meng/data_tmp/box.obj"

    box = BoxBBox()
    box.parse_obj(obj_file_name)
    box.generate_box(save_path)

    print("Done.")









