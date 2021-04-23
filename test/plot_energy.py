import seaborn as sns # for data visualization
import pandas as pd # for data analysis
import matplotlib.pyplot as plt # for data visualization
import numpy as np


if __name__ == "__main__":
	#floor_friction = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
	#pushing_energy = [0.470084, 0.975649, 1.482395, 1.991938, 2.500672, 3.003495, 3.518342, 4.016026, 4.513929]
	#robot_energy = [2.630368, 2.727561, 2.908304, 3.140827, 3.397789, 3.634954, 3.876243, 4.117082, 4.346900]


	floor_friction = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
	pushing_energy = [0.975649, 1.482395, 1.991938, 2.500672, 3.003495, 3.518342, 4.016026]
	robot_energy = [2.727561, 2.908304, 3.140827, 3.397789, 3.634954, 3.876243, 4.117082]
	energy_ratio = np.divide(np.array(pushing_energy), np.array(robot_energy))

	energy_data = pd.DataFrame({
	'friction': floor_friction, 
	'robot energy': robot_energy,
	'pushing energy': pushing_energy})

	energy_ratio_data = pd.DataFrame({
	'friction': floor_friction, 
	'energy ratio': energy_ratio})

	#print(energy_data)
	
	energy_data = pd.melt(energy_data, id_vars=['friction'], var_name="energy_category", value_name="energy")
	print(energy_data)
	
	fig = plt.figure()
	sns.lineplot(x="friction", y='energy', hue='energy_category', data=energy_data, marker="o")
	plt.show()
	fig.savefig('energy_category.png')
	
	fig = plt.figure()
	sns.lineplot(x="friction", y='energy ratio', data=energy_ratio_data, marker="o")
	plt.show()
	fig.savefig('energy_ratio.png')
	
	print("Done.")