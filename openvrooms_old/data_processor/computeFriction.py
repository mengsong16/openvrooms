import cv2
import numpy as np
import scipy.io as io
import os

def compute_friction(albedo_image_path, roughness_image_path, table_file):
	albedo = cv2.imread(albedo_image_path)
	#print(albedo.shape)
	image_size = albedo.shape[:2]

	albedo = albedo[:, :, ::-1]
	albedo = albedo.astype(np.float32 ) / 255.0
	albedo = np.round(albedo ** (2.2 ) * 4 + 0.5 )
	albedo = np.clip(albedo, 0, 4 ).astype(np.int32 )

	r = albedo[:, :, 0].flatten()
	g = albedo[:, :, 1].flatten()
	b = albedo[:, :, 2].flatten()

	rgh = cv2.imread(roughness_image_path)
	#print(rgh.shape)
	rgh = rgh[:, :,0]
	rgh = rgh.astype(np.float32 ) / 255.0
	rgh = np.round(rgh * 10 ).astype(np.int32 )
	rgh = rgh.flatten()

	index = r * 5 * 5 * 11 + g * 5 * 11 + b * 11 + rgh

	frictionCoefMap = np.load(table_file)
	frictionCoefMap = frictionCoefMap.flatten()

	minFric = frictionCoefMap.min()
	maxFric = frictionCoefMap.max()

	frictionIm = frictionCoefMap[index ]
	#frictionIm = frictionIm.reshape(480, 640 )
	frictionIm = frictionIm.reshape(image_size[0], image_size[1])
	frictionIm = (frictionIm - minFric) / (maxFric - minFric )

	mean_friction = np.mean(np.array(frictionIm))

	return mean_friction


	#frictionIm = (frictionIm * 255.0).astype(np.float32)
	#cv2.imwrite(os.path.join(root_dir, 'friction.png'), frictionIm)

def test_case():
	root_dir = "/Users/meng/Documents/2020_Fall/openroom/cvpr2021/svbrdf2friction"
	image_folder = os.path.join(root_dir, "Friction")
	table_file = os.path.join(root_dir, "frictionMap.npy")
	albedo_image_path = os.path.join(image_folder, "imbaseColor_4_2.png")
	roughness_image_path = os.path.join(image_folder, "imroughness_4_2.png")
	compute_friction(albedo_image_path, roughness_image_path, table_file)

if __name__ == "__main__":
	root_dir = "/Users/meng/Documents/iopenroom/dataset/original/BRDFOriginDataset"
	table_file = "/Users/meng/Documents/2020_Fall/openroom/cvpr2021/svbrdf2friction/frictionMap.npy"

	material_list = ['Material__wood_cherry']
	
	albedo_image_name = "tiled/diffuse_tiled.png"
	roughness_image_name = "tiled/rough_tiled.png"

	material_list = ['Material__wood_cherry', 'Material__wax_paint_white', 'Material__fabric_carpet_grey', 'Material__carpet_loop', 'Material__dmls_silver_mirror_polished', 'Material__wood_crate_large']

	for material in material_list:
		albedo_image_path = os.path.join(root_dir, material, albedo_image_name)
		roughness_image_path = os.path.join(root_dir, material, roughness_image_name)
		
		if os.path.exists(albedo_image_path) and os.path.exists(roughness_image_path):
			fric = compute_friction(albedo_image_path, roughness_image_path, table_file)
			print("Material: %s, %f"%(material, fric))
