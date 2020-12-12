import os
from openvrooms.data_processor.adapted_object_urdf import ObjectUrdfBuilder
import os
import pybullet as p
import numpy as np
import time

import sys
#sys.path.insert(0, "../")
#from config import *
from openvrooms.config import *

# Build a URDF from an object file
# center: 'mass' - center of mass, 'geometric' - centroid, 
		  # face - 'top', 'bottom', 'xy_pos', 'xy_neg','xz_pos','xz_neg','yz_pos','yz_neg'
# reset the position x,y,z of geometry and collision in urdf
# input: xyz.obj
# output: xyz.urdf, xyz_vhacd.obj
def generate_urdf(obj_file, object_folder, urdf_prototype_file, force_decompose=True, mass=None, center='geometric'):
	log_file = os.path.join(object_folder, "vhacd_log.txt")
	builder = ObjectUrdfBuilder(object_folder, log_file=log_file, urdf_prototype=urdf_prototype_file)
	builder.build_urdf(filename=obj_file, force_overwrite=True, decompose_concave=True, force_decompose=force_decompose, mass=mass, center=center)

def test_urdf(urdf_file):
	p.connect(p.GUI)
	p.setGravity(0, 0, -9.8)
	p.setTimeStep(1./240.)

	boxStartPos = [0, 0, 0.5]
	boxStartOr = p.getQuaternionFromEuler(np.deg2rad([0, 0, 0]))
	boxId = p.loadURDF(urdf_file, boxStartPos, boxStartOr)

	for _ in range(24000):  # at least 100 seconds
		p.stepSimulation()
		time.sleep(1./240.)

	p.disconnect()

if __name__ == "__main__":
	#root_path = "/Users/meng/Documents/object2urdf/examples/basketball"
	#urdf_prototype_file = os.path.join(root_path, '_prototype_ball.urdf') # urdf template
	#object_folder = os.path.join(root_path, "ball")
	#obj_file = os.path.join(object_folder, "basketball_corrected.obj")	
	#urdf_file = os.path.join(object_folder, "basketball_corrected.urdf")	
	
	urdf_prototype_file = os.path.join(metadata_path, 'urdf_prototype.urdf') # urdf template
	object_folder = os.path.join(interative_dataset_path, 'scene0420_01')
	obj_file = os.path.join(object_folder, "curtain_4_object_alignedNew.obj")	
	urdf_file = os.path.join(object_folder, "curtain_4_object_alignedNew.urdf")	
	
	generate_urdf(obj_file, object_folder, urdf_prototype_file, mass=10)

	#test_urdf(urdf_file)