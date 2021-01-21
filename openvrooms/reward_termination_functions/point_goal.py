from openvrooms.reward_termination_functions.reward_termination_function_base import BaseRewardTerminationFunction
from gibson2.utils.utils import l2_distance
import numpy as np


class PointGoal(BaseRewardTerminationFunction):
    """
    PointGoal used for navigation tasks
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(PointGoal, self).__init__(config)
        self.dist_tol = self.config.get('dist_tol', 0.1)
        self.success_reward = self.config.get('success_reward', 10.0)

        self.use_goal_dist_reward = self.config.get('use_goal_dist_reward', True)
        self.goal_dist_reward_weight = self.config.get('goal_dist_reward_weight', 1.0)

    def reset(self, task, env):
        """
        Compute the initial goal distance after episode reset

        :param task: task instance
        :param env: environment instance
        """
        if self.use_goal_dist_reward:
            self.goal_dist = task.goal_distance()


    def get_reward_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        new_goal_dist = task.goal_distance()

        #print(new_goal_dist)

        # check success / done
        if new_goal_dist > self.dist_tol:
            done = False
        else:
            done = True    

        success = done

        # get success reward
        if success:
            reward = self.success_reward
        else:
            reward = 0.0    
        
        # get goal reward
        if self.use_goal_dist_reward:
            goal_dist_reward = self.goal_dist_reward_weight * (self.goal_dist - new_goal_dist)
            self.goal_dist = new_goal_dist

            reward += goal_dist_reward
   

        return reward, done, success

    def get_name(self):
        return "point_goal"    
