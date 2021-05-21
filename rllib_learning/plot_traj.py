import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from gibson2.utils.utils import parse_config

## dimensions
X_MIN, X_MAX, Y_MIN, Y_MAX = -3.418463, 3.532937, -2.803233, 2.722267
ROBOT_R = 0.25
BOX_W, BOX_L = 0.452909, 0.967061

## colors
COLOR_POOL1 = ['pink', 'orange']
COLOR_POOL2 = ['lightblue', 'lightgreen']

class PlotTrajectory:
    def __init__(self, eval_trial_path: str, ax=None):
        self.eval_trial_path = eval_trial_path
        self.rollouts = self.__read_rollouts()
        self.success_robot_traj, self.success_box_traj, self.failed_robot_traj, self.failed_box_traj = self.__read_traj()
        self.robot_traj_len, self.box_traj_len = self.__get_traj_len()
        self.config = self.__read_config()

        if ax == None:
            self.fig = plt.figure()
            self.ax = self.fig.gca()
        else:
            self.fig = None
            self.ax = ax

    def __read_rollouts(self):
        rollout_file = os.path.join(self.eval_trial_path, 'rollouts.pkl')
        assert os.path.isfile(rollout_file), f"{rollout_file} is not a file!"
        with open(rollout_file, 'rb') as f:
            rollouts = pickle.load(f)
        return rollouts

    def __read_config_file(self) -> str:
        param_file = os.path.join(self.eval_trial_path, 'params.pkl')
        with open(param_file, 'rb') as f:
            params = pickle.load(f)
            config_filename = params['env_config']['config_file']
        config_file = os.path.join(self.eval_trial_path, config_filename)
        return config_file
    
    def __read_config(self):
        config_file = self.__read_config_file()
        if not os.path.isfile(config_file):
            print(f"Warning: {config_file} is not a file! Not able to plot floor bands and start/goal poses!")
            return None
        return parse_config(config_file)
    
    def __read_traj(self):
        # num_success_eps = 0
        # robot_traj_len = list()
        # box_traj_len = list()
        success_robot_traj, failed_robot_traj = list(), list()
        success_box_traj, failed_box_traj = list(), list()

        for eps in self.rollouts:
            # True if 'success' record is True, or no 'success' record
            is_success = (len(eps[-1])!=6 or eps[-1][-1]['success'])

            robot_traj, box_traj = list(), list()
            # robot_x, robot_y = list(), list()
            # box_x, box_y = list(), list()
            for step in eps:
                obs = step[0].reshape((4, 3))
                next_obs = step[2].reshape((4, 3))

                robot_traj.append(obs[0, :2])
                box_traj.append(obs[2, :2])
                # robot_x.append(np.array([obs[0, 0], next_obs[0, 0]]))
                # robot_y.append(np.array([obs[0, 1], next_obs[0, 1]]))
                # box_x.append(np.array([obs[2, 0], next_obs[2, 0]]))
                # box_y.append(np.array([obs[2, 1], next_obs[2, 1]]))
            robot_traj.append(next_obs[0, :2])
            box_traj.append(next_obs[2, :2])

            if is_success:
                success_robot_traj.append(np.array(robot_traj, dtype=float)) # [(N_i, 2)]
                success_box_traj.append(np.array(box_traj, dtype=float)) # [(N_i, 2)]
            else: 
                failed_robot_traj.append(np.array(robot_traj, dtype=float)) # [(N_i, 2)]
                failed_box_traj.append(np.array(box_traj, dtype=float)) # [(N_i, 2)]

            # robot_x, robot_y = np.array(robot_x, dtype=float), np.array(robot_y, dtype=float) # (N, 2)
            # box_x, box_y = np.array(box_x, dtype=float), np.array(box_y, dtype=float) # (N, 2)
        return success_robot_traj, success_box_traj, failed_robot_traj, failed_box_traj
    
    # traj: [[x_i, y_i]], (N, 2)
    def calc_traj_len(self, traj: np.ndarray) -> float:
        return np.sum(np.linalg.norm(traj[1:, :] - traj[:-1, :], axis=1))

    def __get_traj_len(self):
        num_traj = len(self.success_robot_traj)
        if num_traj == 0:
            return list(), list()
        
        # calc trajectory lengths
        robot_traj_len, box_traj_len = list(), list()
        for robot_traj, box_traj in zip(self.success_robot_traj, self.success_box_traj):
            robot_traj_len.append(self.calc_traj_len(robot_traj))
            box_traj_len.append(self.calc_traj_len(box_traj))

        return robot_traj_len, box_traj_len # list, list
    
    def get_traj_stats(self):
        if len(self.robot_traj_len) == 0:
            return 0., 0.
        
        # calc trajectory length mean & std
        robot_mean = np.mean(self.robot_traj_len)
        robot_std = np.std(self.robot_traj_len)
        box_mean = np.mean(self.box_traj_len)
        box_std = np.std(self.box_traj_len)

        return robot_mean, robot_std, box_mean, box_std

    def plot_traj(self, traj_list, color='k', linestyle='-', label='trajectory', zorder=1):
        if len(traj_list) == 0: return
        for traj in traj_list:
            line, = self.ax.plot(traj[:, 0], traj[:, 1], linewidth=1, color=color, linestyle=linestyle, zorder=zorder)
        line.set_label(label)
    
    def plot_shortest_traj(self, color='g', linestyle='-'):
        if self.config == None: return

        robot_start = self.config['agent_initial_pos']
        
        for (box_start, box_goal) in zip(self.config['obj_initial_pos'], self.config['obj_target_pos']):
            ## robot to box
            self.ax.plot([robot_start[0], box_start[0]], [robot_start[1], box_start[1]], color=color, linestyle=linestyle, linewidth=1.5, zorder=10)
            ## box start to box goal
            line, = self.ax.plot([box_start[0], box_goal[0]], [box_start[1], box_goal[1]], color=color, linestyle=linestyle, linewidth=1.5, zorder=10)
        
        line.set_label('shortest path')

    
    def crop_graph(self, xmin=X_MIN, xmax=X_MAX, ymin=Y_MIN, ymax=Y_MAX):
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

    def plot_floor(self, low_friction_color_pool=COLOR_POOL1, high_friction_color_pool=COLOR_POOL2):
        if self.config == None: return

        low_idx = -1
        high_idx = -1
        floor_friction = self.config['floor_friction']
        seen_friction = []

        if 'floor_borders' in self.config:
            if self.config['border_type'] == 'y_border':
                floor_borders = [Y_MIN] + self.config['floor_borders'] + [Y_MAX]
                for i in range(len(floor_borders)-1):
                    f = floor_friction[i]
                    if f < 0.5:
                        if not (f in seen_friction): 
                            low_idx = (low_idx + 1) % len(low_friction_color_pool)
                            seen_friction.append(f)
                        color = low_friction_color_pool[low_idx]
                    else:
                        if not (f in seen_friction): 
                            high_idx = (high_idx + 1) % len(high_friction_color_pool)
                            seen_friction.append(f)
                        color = high_friction_color_pool[high_idx]
                    band = patches.Rectangle((X_MIN, floor_borders[i]), X_MAX-X_MIN, floor_borders[i+1]-floor_borders[i], facecolor=color, alpha=0.5)
                    self.ax.add_patch(band)
            elif config['border_type'] == 'x_border':
                floor_borders = [X_MIN] + config['floor_borders'] + [X_MAX]
                for i in range(len(floor_borders)-1):
                    f = floor_friction[i]
                    if f < 0.5:
                        if not (f in seen_friction): 
                            low_idx = (low_idx + 1) % len(low_friction_color_pool)
                            seen_friction.append(f)
                        color = low_friction_color_pool[low_idx]
                    else:
                        if not (f in seen_friction): 
                            high_idx = (high_idx + 1) % len(high_friction_color_pool)
                            seen_friction.append(f)
                        color = high_friction_color_pool[high_idx]
                    band = patches.Rectangle((floor_borders[i], Y_MIN), floor_borders[i+1]-floor_borders[i], Y_MAX-Y_MIN, facecolor=color, alpha=0.5)
                    self.ax.add_patch(band)
        else:
            if floor_friction[0] < 0.5:
                color = low_friction_color_pool[0]
            else:
                color = high_friction_color_pool[0]
            band = patches.Rectangle((X_MIN, Y_MIN), X_MAX-X_MIN, Y_MAX-Y_MIN, facecolor=color, alpha=0.5)
            self.ax.add_patch(band)
        
    def plot_robot(self, color='r', draw_shape=True):
        if self.config == None: return

        robot_start = self.config['agent_initial_pos']
        self.ax.scatter(robot_start[0], robot_start[1], c=color, s=20, marker='s', zorder=15)
        if draw_shape:
            robot = patches.Circle((robot_start[0], robot_start[1]), radius=ROBOT_R, facecolor=color, alpha=0.5)
            self.ax.add_patch(robot)

    def plot_box(self, color='b', draw_shape=True):
        if self.config == None: return

        r = 0.5 * np.sqrt(BOX_L**2 + BOX_W**2)
        a = np.arctan(BOX_W / BOX_L)
        for (start_pos, start_orn, goal_pos, goal_orn) in zip(self.config['obj_initial_pos'], self.config['obj_initial_orn'], self.config['obj_target_pos'], self.config['obj_target_orn']):
            self.ax.scatter(start_pos[0], start_pos[1], c=color, s=20, marker='s', zorder=15) # box_start
            self.ax.scatter(goal_pos[0], goal_pos[1], c=color, s=20, marker='o', zorder=15) # box_goal
            if draw_shape:
                theta1 = start_orn[-1]
                theta2 = goal_orn[-1]
                delta_x1 = r * np.cos(3*np.pi/2 - a + theta1)
                delta_x2 = r * np.cos(3*np.pi/2 - a + theta2)
                delta_y1 = r * np.sin(3*np.pi/2 - a + theta1)
                delta_y2 = r * np.sin(3*np.pi/2 - a + theta2)
                box1 = patches.Rectangle([start_pos[0]+delta_x1, start_pos[1]+delta_y1], BOX_W, BOX_L, angle=180*theta1/np.pi, facecolor=color, alpha=0.5)
                box2 = patches.Rectangle([goal_pos[0]+delta_x2, goal_pos[1]+delta_y2], BOX_W, BOX_L, angle=180*theta2/np.pi, facecolor=color, alpha=0.5)
                self.ax.add_patch(box1)
                self.ax.add_patch(box2)

    def plot_settings(self):
        self.plot_floor()
        self.plot_robot()
        self.plot_box()

## plot trajectories for 
def plot_trajectory(eval_trial_path: str, save_file: str, robot=True, box=True, success=True, failure=True):
    print("\n=== PLOT TRAJECTORY ===\n")
    print("robot  : ", robot)
    print("box    : ", box)
    print("success: ", success)
    print("failure: ", failure)
    print("\nplotting trajectories for '" + eval_trial_path + "' ...\n")
    plot_traj = PlotTrajectory(eval_trial_path)

    ## get stats
    robot_mean, robot_std, box_mean, box_std = plot_traj.get_traj_stats()

    ## plot trajectories
    if robot:
        if success: plot_traj.plot_traj(plot_traj.success_robot_traj, 'r', '-', 'robot success', 5)
        if failure: plot_traj.plot_traj(plot_traj.failed_robot_traj, 'r', '--', 'robot failure', 5)
    if box:
        if success: plot_traj.plot_traj(plot_traj.success_box_traj, 'b', '-', 'box success', 5)
        if failure: plot_traj.plot_traj(plot_traj.failed_box_traj, 'b', '--', 'box failure', 5)
    plot_traj.plot_shortest_traj()

    ## plot settings
    plot_traj.plot_floor()
    if robot: plot_traj.plot_robot()
    if box: plot_traj.plot_box()

    plot_traj.crop_graph()
    fig = plot_traj.fig
    ax = plot_traj.ax

    ## add texts
    ax.legend(loc='lower right')
    ax.set_title('box traj len: mean=%.2f, std=%.2f \nrobot traj len: mean=%.2f, std=%.2f'%(box_mean, box_std, robot_mean, robot_std), fontsize=10)

    fig.savefig(save_file)
    plt.close('all')
    print("\nfile saved to " + save_file)

## environment settings from 'eval_trial_path1'
def compare_trajectory(eval_trial_path1: str, eval_trial_path2: str, save_file: str, robot=False, box=True, success=True, failure=False):
    print("\n=== COMPARE TRAJECTORY ===\n")
    print("robot  : ", robot)
    print("box    : ", box)
    print("success: ", success)
    print("failure: ", failure)

    print("\nplotting trajectories for '" + eval_trial_path1 + "' ...\n")
    plot_traj1 = PlotTrajectory(eval_trial_path1)

    ## get stats
    robot_mean1, robot_std1, box_mean1, box_std1 = plot_traj1.get_traj_stats()

    ## plot trajectories
    plot_traj1.plot_shortest_traj()
    if robot:
        if success: plot_traj1.plot_traj(plot_traj1.success_robot_traj, 'r', '-', 'robot success 1', 5)
        if failure: plot_traj1.plot_traj(plot_traj1.failed_robot_traj, 'r', '--', 'robot failure 1', 5)
    if box:
        if success: plot_traj1.plot_traj(plot_traj1.success_box_traj, 'b', '-', 'box success 1', 5)
        if failure: plot_traj1.plot_traj(plot_traj1.failed_box_traj, 'b', '--', 'box failure 1', 5)

    ## plot settings
    plot_traj1.plot_floor()
    if robot: plot_traj1.plot_robot()
    if box: plot_traj1.plot_box()

    plot_traj1.crop_graph()
    fig = plot_traj1.fig
    ax = plot_traj1.ax

    print("\nplotting trajectories for '" + eval_trial_path2 + "' ...\n")
    plot_traj2 = PlotTrajectory(eval_trial_path2, ax=ax)

    ## get stats
    robot_mean2, robot_std2, box_mean2, box_std2 = plot_traj2.get_traj_stats()

    ## plot trajectories
    if robot:
        if success: line = plot_traj2.plot_traj(plot_traj2.success_robot_traj, 'darkorange', '-', 'robot success 2', 5)
        if failure: line = plot_traj2.plot_traj(plot_traj2.failed_robot_traj, 'darkorange', '--', 'robot failure 2', 5)
    if box:
        if success: line = plot_traj2.plot_traj(plot_traj2.success_box_traj, 'darkviolet', '-', 'box success 2', 5)
        if failure: line = plot_traj2.plot_traj(plot_traj2.failed_box_traj, 'darkviolet', '--', 'box failure 2', 5)

    ## add texts
    ax.legend(loc='lower right')
    ax.set_title('traj len 1 | box: %.2f(%.2f) | robot %.2f(%.2f) \ntraj len 2 | box: %.2f(%.2f) | robot %.2f(%.2f)'%(box_mean1, box_std1, robot_mean1, robot_std1, box_mean2, box_std2, robot_mean2, robot_std2), fontsize=10)

    fig.savefig(save_file)
    plt.close('all')
    print("\nfile saved to " + save_file)

if __name__ == '__main__':
    #eval_trial_path1 = os.path.join('/home/meng/ray_results/PPO', 'PPO_OpenRoomEnvironmentRLLIB_8a7b3_00000_0_2021-05-19_22-23-42')
    #eval_trial_path2 = os.path.join('/home/meng/ray_results/PPO', 'PPO_OpenRoomEnvironmentRLLIB_284d4_00000_0_2021-05-20_09-33-50')
    eval_trial_path2 = os.path.join('/home/meng/ray_results/PPO', 'PPO_OpenRoomEnvironmentRLLIB_193d8_00000_0_2021-05-20_15-45-39')
    save_file = '2band-short-no-energy-100-2.png'
    plot_trajectory(eval_trial_path2, save_file, failure=False)
    #compare_trajectory(eval_trial_path1, eval_trial_path2, save_file)