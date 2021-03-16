import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import os.path as osp
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter


def load_data(data_plot_path, trials):
	data_path = os.path.join(data_plot_path)
	data1 = load_seed_data(os.path.join(data_path, trials[0]))
	data2 = load_seed_data(os.path.join(data_path, trials[1]))
	data3 = load_seed_data(os.path.join(data_path, trials[2]))

	# n*3
	results = pd.concat([data1, data2, data3], axis=1)
	#index = 4000 * results.shape[0]

	# change index from iteration to timesteps
	#old_index = results.index
	#new_index = old_index * 4000
	#print(new_index)
	#results = results.reindex(new_index)
	#print(results)

	return results

def load_seed_data(data_path):
	df = pd.read_csv(os.path.join(data_path, "progress.csv"), usecols = ['episode_reward_mean'])
	#df = sns.load_dataset(os.path.join(data_path, "progress.csv"))
	#df = np.array(df)

	#print(type(df))

	return df
	

def plot_data(data_plot_path, trials):
	# set seaborn
	sns.set(style="darkgrid")	
	# Load the data
	results = load_data(data_plot_path, trials) 
	#print(results.shape)
	fig = plt.figure()

	# Plot each line
	# (may want to automate this part e.g. with a loop).
	sns_plot = sns.lineplot(data=results, legend=False)

	# change index from iteration to timesteps
	start = 0
	end = len(results) + 1
	x_ticks = list(np.arange(start, end, 25))
	#print(x_ticks)
	timesteps_per_iteration = 4000
	max_timestep = end * timesteps_per_iteration

	#plt.ticklabel_format(style='sci', axis='x')
	
	sns_plot.xaxis.set_ticks(x_ticks)
	#sns_plot.xaxis.set_ticklabels([str(tick*timesteps_per_iteration) for tick in x_ticks])
	#plt.ticklabel_format(axis='x', style='sci')
	#sns_plot.xaxis.set_major_formatter(FormatStrFormatter('%.2E'))
	#sns_plot.xaxis.set_ticklabels(["%.2e"%(tick*timesteps_per_iteration) for tick in x_ticks])
	sns_plot.xaxis.set_ticklabels([tick*timesteps_per_iteration/1000000.0 for tick in x_ticks])


	# set x range
	plt.xlim((0,end))
	#plt.xtickformat('%.1f')
	
	# Our y−axis is ”success rate” here.
	plt.ylabel("mean return", fontsize=10)
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
	fig.savefig(os.path.join(data_plot_path, "train_curve_plot.png"))

if __name__ == "__main__":
	#plot_data()
	data_base_path = "/home/meng/ray_results"
	data_plot_path = os.path.join(data_base_path, "PPO_one_box_state_no_energy")
	#data_plot_path = os.path.join(data_base_path, "PPO_one_box_state_with_energy")

	plot_data(data_plot_path, trials=["PPO_OpenRoomEnvironmentRLLIB_525ea_00000_0_2021-03-15_12-12-01", "PPO_OpenRoomEnvironmentRLLIB_a0b12_00000_0_2021-03-15_12-14-13", "PPO_OpenRoomEnvironmentRLLIB_953d6_00000_0_2021-03-15_13-39-47"])
	#plot_data(data_plot_path, trials=["PPO_OpenRoomEnvironmentRLLIB_c1b2b_00000_0_2021-03-15_14-16-49", "PPO_OpenRoomEnvironmentRLLIB_a7297_00000_0_2021-03-15_14-16-05", "PPO_OpenRoomEnvironmentRLLIB_718f2_00000_0_2021-03-15_15-54-48"])

