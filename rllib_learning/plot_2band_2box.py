import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

import pickle
from gibson2.utils.utils import parse_config

def load_data(data_base_path, trials, column_name):
	curves = []
	if isinstance(trials, list):
		for i in range(len(trials)):
			curves.append(load_seed_data(os.path.join(data_base_path[i], trials[i]), column_name=column_name))
	else:
		curves.append(load_seed_data(os.path.join(data_base_path, trials), column_name=column_name))
	# n*3
	results = pd.concat(curves, axis=1)
	#index = 4000 * results.shape[0]

	# change index from iteration to timesteps
	#old_index = results.index
	#new_index = old_index * 4000
	#print(new_index)
	#results = results.reindex(new_index)
	#print(results)

	return results

def load_seed_data(data_path, column_name):
	df = pd.read_csv(os.path.join(data_path, "progress.csv"), usecols=[column_name])
	#df = sns.load_dataset(os.path.join(data_path, "progress.csv"))
	#df = np.array(df)

	#print(type(df))

	return df
	

def plot_data(plot_save_path, data_base_path, curve_groups, column_name, end_iteration, plot_name):
	# set seaborn
	sns.set(style="darkgrid")
	fig = plt.figure()

	# Load the data
	colors = ['r', 'b', 'm', 'g']
	#sns.set_palette("tab10")
	for i in range(len(curve_groups)):
		results = load_data(data_base_path, curve_groups[i], column_name) 
		# Plot each line
		# (may want to automate this part e.g. with a loop).
		sns_plot = sns.lineplot(data=results, legend=False, palette=[colors[i]])



	# change index from iteration to timesteps
	start_iteration = 0
	#end = len(results) + 1
	x_ticks = list(np.arange(start_iteration, end_iteration+1, 25))
	#print(x_ticks)
	timesteps_per_iteration = 4000
	max_timestep = end_iteration * timesteps_per_iteration

	#plt.ticklabel_format(style='sci', axis='x')
	
	sns_plot.xaxis.set_ticks(x_ticks)
	#sns_plot.xaxis.set_ticklabels([str(tick*timesteps_per_iteration) for tick in x_ticks])
	#plt.ticklabel_format(axis='x', style='sci')
	#sns_plot.xaxis.set_major_formatter(FormatStrFormatter('%.2E'))
	#sns_plot.xaxis.set_ticklabels(["%.2e"%(tick*timesteps_per_iteration) for tick in x_ticks])
	sns_plot.xaxis.set_ticklabels([tick*timesteps_per_iteration/1000000.0 for tick in x_ticks])


	# set x range
	plt.xlim((0,end_iteration))
	#plt.xtickformat('%.1f')
	
	# Our y−axis is ”success rate” here.
	if '/' not in column_name:
		y_axis_name = column_name
	else:
		start_index = column_name.find('/')+1
		y_axis_name = column_name[start_index:]	

	y_axis_name = y_axis_name.replace('_', " ")
	y_axis_name = y_axis_name.replace('mean', "")

	plt.ylabel(y_axis_name, fontsize=10)
	# Our x−axis is iteration number.
	plt.xlabel("million steps", fontsize=10)
	
	#plt.xlabel("Iterations", fontsize=10)
	# Our task is called ”Awesome Robot Performance”
	#plt.title("Push One Box", fontsize=12)
	# Legend
	#plt.legend(loc='lower right')
	# Show the plot on the screen
	plt.show()

	# save figure
	fig = sns_plot.get_figure()
	fig.savefig(os.path.join(plot_save_path, plot_name))

def plot_2box(ray_path, plot_save_path, average_seeds=True):
	two_box_data_base_path = [os.path.join(ray_path, "PPO_two_box_one_circle_success1"), os.path.join(ray_path, "PPO_two_box_one_circle_success2")]
	no_swap_no_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_6a57c_00000_0_2021-05-10_12-10-42', 'PPO_OpenRoomEnvironmentRLLIB_25a19_00000_0_2021-05-11_09-58-45']
	swap_no_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_9b13f_00000_0_2021-05-10_14-13-45', 'PPO_OpenRoomEnvironmentRLLIB_58e11_00000_0_2021-05-11_10-07-20']
	no_swap_with_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_ee9ba_00000_0_2021-05-10_14-01-47', 'PPO_OpenRoomEnvironmentRLLIB_47df1_00000_0_2021-05-11_11-04-08']
	swap_with_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_12a58_00000_0_2021-05-10_17-23-13', 'PPO_OpenRoomEnvironmentRLLIB_d0a78_00000_0_2021-05-11_11-29-26']

	if not os.path.exists(plot_save_path):
		os.makedirs(plot_save_path)
	
	if average_seeds:
		data = []
		data.append(no_swap_no_energy_trials)
		data.append(swap_no_energy_trials)
		data.append(no_swap_with_energy_trials)
		data.append(swap_with_energy_trials)
		
		plot_data(plot_save_path, two_box_data_base_path, curve_groups=data, column_name='custom_metrics/episode_pushing_energy_mean', end_iteration=100, plot_name="train_curve_2box_pushing_energy.png")
		plot_data(plot_save_path, two_box_data_base_path, curve_groups=data, column_name='custom_metrics/success_rate_mean', end_iteration=100, plot_name="train_curve_2box_success_rate.png")
	else:
		seed_num = len(no_swap_no_energy_trials)
		for i in range(seed_num):
			data = []
			data.append(no_swap_no_energy_trials[i])
			data.append(swap_no_energy_trials[i])
			data.append(no_swap_with_energy_trials[i])
			data.append(swap_with_energy_trials[i])
			
			plot_data(plot_save_path, two_box_data_base_path[i], curve_groups=data, column_name='custom_metrics/episode_pushing_energy_mean', end_iteration=100, plot_name="train_curve_2box_pushing_energy_%d.png"%(i+1))
			plot_data(plot_save_path, two_box_data_base_path[i], curve_groups=data, column_name='custom_metrics/success_rate_mean', end_iteration=100, plot_name="train_curve_2box_success_rate_%d.png"%(i+1))

def plot_2band(ray_path, plot_save_path, average_seeds=True):
	two_band_data_base_path = [os.path.join(ray_path, "PPO_two_band_success1")]
	high_friction_no_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_56068_00000_0_2021-04-18_13-12-01']
	low_friction_no_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_481f6_00000_0_2021-04-18_22-29-58']
	two_band_no_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_e5ce7_00000_0_2021-04-18_18-02-22']
	two_band_with_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_2b61b_00000_0_2021-04-19_10-10-41']

	if not os.path.exists(plot_save_path):
		os.makedirs(plot_save_path)
		
	if average_seeds:
		data = []
		data.append(high_friction_no_energy_trials)
		data.append(low_friction_no_energy_trials)
		data.append(two_band_no_energy_trials)
		data.append(two_band_with_energy_trials)
		
		plot_data(plot_save_path, two_band_data_base_path, curve_groups=data, column_name='custom_metrics/episode_pushing_energy_mean', end_iteration=200, plot_name="train_curve_2band_pushing_energy.png")
		plot_data(plot_save_path, two_band_data_base_path, curve_groups=data, column_name='custom_metrics/success_rate_mean', end_iteration=200, plot_name="train_curve_2band_success_rate.png")
	else:
		seed_num = len(high_friction_no_energy_trials)
		for i in range(seed_num):
			data = []
			data.append(high_friction_no_energy_trials[i])
			data.append(low_friction_no_energy_trials[i])
			data.append(two_band_no_energy_trials[i])
			data.append(two_band_with_energy_trials[i])
			
			plot_data(plot_save_path, two_band_data_base_path[i], curve_groups=data, column_name='custom_metrics/episode_pushing_energy_mean', end_iteration=200, plot_name="train_curve_2band_pushing_energy_%d.png"%(i+1))
			plot_data(plot_save_path, two_band_data_base_path[i], curve_groups=data, column_name='custom_metrics/success_rate_mean', end_iteration=200, plot_name="train_curve_2band_success_rate_%d.png"%(i+1))


if __name__ == "__main__":
	ray_path = "/home/meng/ray_results"
	plot_save_path = os.path.join(ray_path, "plots")
	#plot_2box(ray_path=ray_path, plot_save_path=os.path.join(plot_save_path, "2box"), average_seeds=False)
	plot_2band(ray_path=ray_path, plot_save_path=os.path.join(plot_save_path, "2band"), average_seeds=False)

