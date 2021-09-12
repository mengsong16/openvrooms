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
		sns_plot = sns.lineplot(data=results, legend=False, palette=[colors[i]], ci="sd")



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

# def plot_2box(ray_path, plot_save_path, average_seeds=True):
# 	two_box_data_base_path = [os.path.join(ray_path, "PPO_two_box_one_circle_success1"), os.path.join(ray_path, "PPO_two_box_one_circle_success2"), ]
# 	no_swap_no_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_6a57c_00000_0_2021-05-10_12-10-42', 'PPO_OpenRoomEnvironmentRLLIB_25a19_00000_0_2021-05-11_09-58-45']
# 	swap_no_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_9b13f_00000_0_2021-05-10_14-13-45', 'PPO_OpenRoomEnvironmentRLLIB_58e11_00000_0_2021-05-11_10-07-20']
# 	no_swap_with_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_ee9ba_00000_0_2021-05-10_14-01-47', 'PPO_OpenRoomEnvironmentRLLIB_47df1_00000_0_2021-05-11_11-04-08']
# 	swap_with_energy_trials=['PPO_OpenRoomEnvironmentRLLIB_12a58_00000_0_2021-05-10_17-23-13', 'PPO_OpenRoomEnvironmentRLLIB_d0a78_00000_0_2021-05-11_11-29-26']

# 	if not os.path.exists(plot_save_path):
# 		os.makedirs(plot_save_path)
	
# 	if average_seeds:
# 		data = []
# 		data.append(no_swap_no_energy_trials)
# 		data.append(swap_no_energy_trials)
# 		data.append(no_swap_with_energy_trials)
# 		data.append(swap_with_energy_trials)
		
# 		#plot_data(plot_save_path, two_box_data_base_path, curve_groups=data, column_name='custom_metrics/episode_pushing_energy_mean', end_iteration=100, plot_name="train_curve_2box_pushing_energy.png")
# 		plot_data(plot_save_path, two_box_data_base_path, curve_groups=data, column_name='custom_metrics/succeed_episode_pushing_energy_mean', end_iteration=100, plot_name="train_curve_2box_pushing_energy.png")
# 		plot_data(plot_save_path, two_box_data_base_path, curve_groups=data, column_name='custom_metrics/success_rate_mean', end_iteration=100, plot_name="train_curve_2box_success_rate.png")
# 	else:
# 		seed_num = len(no_swap_no_energy_trials)
# 		for i in range(seed_num):
# 			data = []
# 			data.append(no_swap_no_energy_trials[i])
# 			data.append(swap_no_energy_trials[i])
# 			data.append(no_swap_with_energy_trials[i])
# 			data.append(swap_with_energy_trials[i])
			
# 			plot_data(plot_save_path, two_box_data_base_path[i], curve_groups=data, column_name='custom_metrics/succeed_episode_pushing_energy_mean', end_iteration=100, plot_name="train_curve_2box_pushing_energy_%d.png"%(i+1))
# 			plot_data(plot_save_path, two_box_data_base_path[i], curve_groups=data, column_name='custom_metrics/success_rate_mean', end_iteration=100, plot_name="train_curve_2box_success_rate_%d.png"%(i+1))

def plot_2band(two_band_data_base_path, plot_save_path, average_seeds=True):
	#two_band_data_base_path = [os.path.join(ray_path, "PPO"), os.path.join(ray_path, "PPO"), os.path.join(ray_path, "PPO")]
	two_band_data_base_path = [two_band_data_base_path, two_band_data_base_path, two_band_data_base_path]
	two_band_no_energy_trials=["PPO_OpenRoomEnvironmentRLLIB_95cd0_00000_0_2021-05-26_18-05-10", "PPO_OpenRoomEnvironmentRLLIB_3fbfc_00000_0_2021-05-27_10-16-18", "PPO_OpenRoomEnvironmentRLLIB_f155f_00000_0_2021-05-27_17-59-23"]
	two_band_with_energy_trials=["PPO_OpenRoomEnvironmentRLLIB_b2d9d_00000_0_2021-05-26_18-05-59", "PPO_OpenRoomEnvironmentRLLIB_5f748_00000_0_2021-05-27_10-17-11", "PPO_OpenRoomEnvironmentRLLIB_12158_00000_0_2021-05-27_20-02-00"]

	if not os.path.exists(plot_save_path):
		os.makedirs(plot_save_path)
		
	if average_seeds:
		data = []
		data.append(two_band_no_energy_trials)
		data.append(two_band_with_energy_trials)
		
		plot_data(plot_save_path, two_band_data_base_path, curve_groups=data, column_name='custom_metrics/succeed_episode_pushing_energy_mean', end_iteration=300, plot_name="train_curve_2band_pushing_energy.png")
		plot_data(plot_save_path, two_band_data_base_path, curve_groups=data, column_name='custom_metrics/success_rate_mean', end_iteration=250, plot_name="train_curve_2band_success_rate.png")
	else:
		seed_num = len(two_band_no_energy_trials)
		for i in range(seed_num):
			data = []
			data.append(two_band_no_energy_trials[i])
			data.append(two_band_with_energy_trials[i])
			
			plot_data(plot_save_path, two_band_data_base_path[i], curve_groups=data, column_name='custom_metrics/succeed_episode_pushing_energy_mean', end_iteration=300, plot_name="train_curve_2band_pushing_energy_%d.png"%(i+1))
			plot_data(plot_save_path, two_band_data_base_path[i], curve_groups=data, column_name='custom_metrics/success_rate_mean', end_iteration=250, plot_name="train_curve_2band_success_rate_%d.png"%(i+1))

def sort_trials(base_path):
	trials = [None for i in range(4)]
	config_num = None

	for fname in os.listdir(base_path):
		fpath = os.path.join(base_path, fname)
		# folder
		if os.path.isdir(fpath):
			# get config file name
			param_file = os.path.join(fpath, 'params.pkl')
			with open(param_file, 'rb') as f:
				params = pickle.load(f)
				config_filename = params['env_config']['config_file']
			config_file = os.path.join(fpath, config_filename)
			# get config file
			config = parse_config(config_file)
			# check config_num
			if config_num == None:
				config_num == config['config_index']
			else:
				assert config_num == config['config_index'], f"[sort_trials]Error: config_index of '{fname}' doesn't match: {config['config_index']} v.s. {config_num}!"
			# sort
			idx = 0
			if config['swap']: idx += 1
			if config['use_energy_cost']: idx += 2
			trials[idx] = fname

	# check
	assert not (None in trials), f"[sort_trials]Error: incomplete set of trials, trials got: {trials}!"

	return trials

def plot_2box(ray_path, plot_save_path, average_seeds=True):
	two_box_data_base_path = [os.path.join(ray_path, "config0"), os.path.join(ray_path, "config2") , os.path.join(ray_path, "config3"), os.path.join(ray_path, "config4"), os.path.join(ray_path, "config7")]

	#print(two_box_data_base_path)
	# exit()
	# no_swap_no_energy_trials   = ['PPO_OpenRoomEnvironmentRLLIB_d2b62_00000_0_2021-05-25_05-43-36', 'PPO_OpenRoomEnvironmentRLLIB_291d0_00000_0_2021-05-16_20-58-38', 'PPO_OpenRoomEnvironmentRLLIB_ae0e5_00000_0_2021-05-17_00-29-57', 'PPO_OpenRoomEnvironmentRLLIB_1d57e_00000_0_2021-05-25_11-14-58', 'PPO_OpenRoomEnvironmentRLLIB_23fc6_00000_0_2021-05-16_21-34-17']
	# swap_no_energy_trials      = ['PPO_OpenRoomEnvironmentRLLIB_051a6_00000_0_2021-05-25_05-45-01', 'PPO_OpenRoomEnvironmentRLLIB_93aa1_00000_0_2021-05-16_21-01-37', 'PPO_OpenRoomEnvironmentRLLIB_487a3_00000_0_2021-05-17_00-41-25', 'PPO_OpenRoomEnvironmentRLLIB_caf90_00000_0_2021-05-25_11-12-40', 'PPO_OpenRoomEnvironmentRLLIB_eb3f6_00000_0_2021-05-16_22-01-20']
	# no_swap_with_energy_trials = ['PPO_OpenRoomEnvironmentRLLIB_7fbae_00000_0_2021-05-25_05-05-29', 'PPO_OpenRoomEnvironmentRLLIB_79c20_00000_0_2021-05-25_02-27-50', 'PPO_OpenRoomEnvironmentRLLIB_2d7e7_00000_0_2021-05-25_04-05-55', 'PPO_OpenRoomEnvironmentRLLIB_a4fff_00000_0_2021-05-25_14-10-34', 'PPO_OpenRoomEnvironmentRLLIB_041bf_00000_0_2021-05-25_03-14-40']
	# swap_with_energy_trials    = ['PPO_OpenRoomEnvironmentRLLIB_9b768_00000_0_2021-05-25_05-06-16', 'PPO_OpenRoomEnvironmentRLLIB_6aa9b_00000_0_2021-05-25_01-58-47', 'PPO_OpenRoomEnvironmentRLLIB_4a0c3_00000_0_2021-05-25_04-06-43', 'PPO_OpenRoomEnvironmentRLLIB_d8443_00000_0_2021-05-25_14-12-00', 'PPO_OpenRoomEnvironmentRLLIB_e3f23_00000_0_2021-05-25_02-59-27']

	# sort trials
	no_swap_no_energy_trials, swap_no_energy_trials, no_swap_with_energy_trials, swap_with_energy_trials = list(), list(), list(), list()
	for base_path in two_box_data_base_path:
		trials = sort_trials(base_path)
		no_swap_no_energy_trials.append(   trials[0])
		swap_no_energy_trials.append(      trials[1])
		no_swap_with_energy_trials.append( trials[2])
		swap_with_energy_trials.append(    trials[3])
	
	if not os.path.exists(plot_save_path):
		os.makedirs(plot_save_path)

	two_box_data_base_path += two_box_data_base_path
	no_energy_trials = no_swap_no_energy_trials+swap_no_energy_trials
	with_energy_trials = no_swap_with_energy_trials+swap_with_energy_trials

	#print(no_energy_trials)
	#print(with_energy_trials)

	if average_seeds:
		data = []
		# data.append(no_swap_no_energy_trials)
		# data.append(swap_no_energy_trials)
		# data.append(no_swap_with_energy_trials)
		# data.append(swap_with_energy_trials)
		data.append(no_energy_trials)
		data.append(with_energy_trials)
		
		#plot_data(plot_save_path, two_box_data_base_path, curve_groups=data, column_name='custom_metrics/episode_pushing_energy_mean', end_iteration=100, plot_name="train_curve_2box_pushing_energy.png")
		plot_data(plot_save_path, two_box_data_base_path, curve_groups=data, column_name='custom_metrics/succeed_episode_pushing_energy_mean', end_iteration=100, plot_name="train_curve_2box_pushing_energy.png")
		plot_data(plot_save_path, two_box_data_base_path, curve_groups=data, column_name='custom_metrics/success_rate_mean', end_iteration=100, plot_name="train_curve_2box_success_rate.png")
	else:
		# seed_num = len(no_swap_no_energy_trials)
		seed_num = len(no_energy_trials)
		for i in range(seed_num):
			data = []
			# data.append(no_swap_no_energy_trials[i])
			# data.append(swap_no_energy_trials[i])
			# data.append(no_swap_with_energy_trials[i])
			# data.append(swap_with_energy_trials[i])
			data.append(no_energy_trials[i])
			data.append(with_energy_trials[i])
			
			plot_data(plot_save_path, two_box_data_base_path[i], curve_groups=data, column_name='custom_metrics/succeed_episode_pushing_energy_mean', end_iteration=100, plot_name="train_curve_2box_pushing_energy_%d.png"%(i+1))
			plot_data(plot_save_path, two_box_data_base_path[i], curve_groups=data, column_name='custom_metrics/success_rate_mean', end_iteration=100, plot_name="train_curve_2box_success_rate_%d.png"%(i+1))

if __name__ == "__main__":
	ray_path = "/home/meng/ray_results"
	plot_save_path = os.path.join(ray_path, "plots")

	# plot 2 band results
	plot_2band(two_band_data_base_path=os.path.join(ray_path, "two-band-paper"), plot_save_path=os.path.join(plot_save_path, "2band"), average_seeds=False)
	print("2 band separate plot Done.")
	plot_2band(two_band_data_base_path=os.path.join(ray_path, "two-band-paper"), plot_save_path=os.path.join(plot_save_path, "2band"), average_seeds=True)
	print("2 band overall plot Done.")

	# plot 2 box results
	plot_2box(ray_path=os.path.join(ray_path, "two-box-paper"), plot_save_path=os.path.join(plot_save_path, "2box"), average_seeds=False)
	print("2 box separate plot Done.")
	plot_2box(ray_path=os.path.join(ray_path, "two-box-paper"), plot_save_path=os.path.join(plot_save_path, "2box"), average_seeds=True)
	print("2 box overall plot Done.")
