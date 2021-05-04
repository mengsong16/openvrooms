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

X_MIN, X_MAX, Y_MIN, Y_MAX = -3.418463, 3.532937, -2.803233, 2.722267
def plot_trajectory(eval_trial_path: str, config_filename: str, plot_robot: False, plot_success: True):
	print("\n=== Plot Trajectories ===\n")
	## read rollout records
	rollout_file = os.path.join(eval_trial_path, 'rollouts.pkl')
	assert os.path.isfile(rollout_file), f"{rollout_file} is not a file!"
	with open(rollout_file, 'rb') as f:
		data = pickle.load(f)

	num_eps = len(data)
	if len(data[0][0]) == 5: 
		print("Warning: NO success info recorded!")
		plot_success = False

	fig = plt.figure()
	ax = fig.gca()

	## read env config, plot start/goal positions
	config_file = os.path.join(eval_trial_path, config_filename)
	if os.path.isfile(config_file): 
		config = parse_config(config_file)
		robot_start = config['agent_initial_pos']
		box_start = config['obj_initial_pos'][0]
		box_goal = config['obj_target_pos'][0]
		ax.scatter(box_start[0], box_start[1], c='blue', s=20, marker='s', zorder=10) # box_start
		ax.scatter(box_goal[0], box_goal[1], c='blue', s=20, marker='o', zorder=10) # box_goal
		if plot_robot:
			ax.scatter(robot_start[0], robot_start[1], c='red', s=20, marker='s', zorder=10) # robot_start
	else:
		print(f"Warning: {config_file} is not a file! No start/goal position plotted!")
	
	# plot shortest path
	ax.plot([box_start[0], box_goal[0]], [box_start[1], box_goal[1]], 'g-', linewidth=1, zorder=5)

	## plot trajectories
	num_success_eps = 0
	for eps in data:
		is_success = eps[-1][-1]['success']

		# skip for unsuccessful episode
		if plot_success and not is_success: 
			continue

		# solid trajectory: success
		# dashed trajectory: failure
		if is_success: 
			line_format = '-'
			num_success_eps += 1
		else:
			line_format = '--'

		robot_x, robot_y = list(), list()
		box_x, box_y = list(), list()
		for step in eps:
			obs = step[0].reshape((4, 3))
			next_obs = step[0].reshape((4, 3))

			robot_x.append(np.array([obs[0, 0], next_obs[0, 0]]))
			robot_y.append(np.array([obs[0, 1], next_obs[0, 1]]))
			box_x.append(np.array([obs[2, 0], next_obs[2, 0]]))
			box_y.append(np.array([obs[2, 1], next_obs[2, 1]]))

		# plot box trajectory
		ax.plot(np.array(box_x), np.array(box_y), 'b'+line_format, linewidth=1)

		# plot robot trajectory
		if plot_robot:
			ax.plot(np.array(robot_x), np.array(robot_y), 'r'+line_format, linewidth=1)
		
	# crop graph to size of room
	ax.set_xlim(X_MIN, X_MAX)
	ax.set_ylim(Y_MIN, Y_MAX)

	# save figure
	fig.savefig(f"{os.path.join(eval_trial_path, 'trajectories.png')}")
	plt.close('all')

	# summary
	if plot_success: 
		print("\nOnly plot successful trajectories: %d / %d"%(num_success_eps, num_eps))
	else:
		print("\nPlot success(solid[%d]) + failure(dashed[%d]) trajectories: %d"%(num_success_eps, num_eps-num_success_eps, num_eps))
	print("---------")
	if plot_robot:
		print("Plot trajectories for robot(red) and box(blue)\n")
	else:
		print("Plot trajectories for only box\n")
	


# if __name__ == "__main__":
# 	#plot_data()
# 	data_base_path = "/home/meng/ray_results"
# 	data_plot_path = os.path.join(data_base_path, "PPO_one_box_state_no_energy")
# 	#data_plot_path = os.path.join(data_base_path, "PPO_one_box_state_with_energy")

# 	plot_data(data_plot_path, trials=["PPO_OpenRoomEnvironmentRLLIB_525ea_00000_0_2021-03-15_12-12-01", "PPO_OpenRoomEnvironmentRLLIB_a0b12_00000_0_2021-03-15_12-14-13", "PPO_OpenRoomEnvironmentRLLIB_953d6_00000_0_2021-03-15_13-39-47"])
# 	#plot_data(data_plot_path, trials=["PPO_OpenRoomEnvironmentRLLIB_c1b2b_00000_0_2021-03-15_14-16-49", "PPO_OpenRoomEnvironmentRLLIB_a7297_00000_0_2021-03-15_14-16-05", "PPO_OpenRoomEnvironmentRLLIB_718f2_00000_0_2021-03-15_15-54-48"])

if __name__ == "__main__":
	eval_trial_path = os.path.join('/home/yuhan/ray_results/PPO/', 'PPO_OpenRoomEnvironmentRLLIB_99ba9_00000_0_2021-04-30_19-54-51')
	config_filename = 'fetch_relocate_multi_band.yaml'
	plot_trajectory(eval_trial_path, config_filename, plot_robot=True, plot_success=False)