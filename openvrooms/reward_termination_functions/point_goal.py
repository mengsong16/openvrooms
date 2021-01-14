from openvrooms.reward_termination_functions.reward_termination_function_base import BaseRewardTerminationFunction
from gibson2.utils.utils import l2_distance
import numpy as np


class PointGoal(BaseRewardTerminationFunction):
    """
    PointGoal used for PointNavFixed/RandomTask
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(PointGoal, self).__init__(config)
        self.dist_tol = self.config.get('dist_tol', 0.1)
        self.angle_tol = self.config.get('angle_tol', 0.2)
        self.use_goal_dist_reward = self.config.get('use_goal_dist_reward', True)
        self.rot_dist_reward_weight = self.config.get('rot_dist_reward_weight', 0.2)
        self.success_reward = self.config.get('success_reward', 10.0)

    def get_reward_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        pos_distances, rot_distances = task.goal_distance()

        #print(pos_distances)
        #print(rot_distances)

        assert pos_distances.shape[0] == rot_distances.shape[0] == task.obj_num

        # check success / done
        done = True
        for i in list(range(task.obj_num)):
            if pos_distances[i] > self.dist_tol or rot_distances[i] > self.angle_tol:
                done = False
                break

        success = done

        # get success reward
        if success:
            reward = self.success_reward
        else:
            reward = 0.0    
        # get goal reward
        if self.use_goal_dist_reward:
            reward += (np.mean(pos_distances) + self.rot_dist_reward_weight * np.mean(rot_distances))
   

        return reward, done, success

    def get_name(self):
        return "point_goal"    
