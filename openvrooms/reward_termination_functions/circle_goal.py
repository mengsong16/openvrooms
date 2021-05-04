from openvrooms.reward_termination_functions.reward_termination_function_base import BaseRewardTerminationFunction
from gibson2.utils.utils import l2_distance
import numpy as np


class CircleGoal(BaseRewardTerminationFunction):
    """
    ObjectGoal used for object relocation tasks
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(CircleGoal, self).__init__(config)
        self.success_reward = float(self.config.get('success_reward', 10.0))

        self.success = False
        self.goal_object = 0
        self.obj_num = None

    def reset(self, task, env):
        """
        Compute the initial goal distance after episode reset

        :param task: task instance
        :param env: environment instance
        """
        self.success = False
        self.goal_object = 0 
        self.obj_num = task.obj_num   


    # step
    def get_reward_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        pos_distances = task.circle_goal_distance()
        sorted_pos_distances = list(np.sort(pos_distances))
        sorted_radius =  list(np.sort(task.circle_radius))

        # count how many objects are in the goal position
        self.goal_object = 0

        # circle_num == obj_num   
        for i in list(range(len(sorted_pos_distances)-1)):
            if sorted_pos_distances[i] > sorted_radius[i] and sorted_pos_distances[i] <= sorted_radius[i+1]:
                self.goal_object += 1           
    
        # check the last one
        if sorted_pos_distances[-1] > sorted_radius[-1]:
            self.goal_object += 1

        # succeed
        if self.goal_object == self.obj_num:
            done = True
        else:
            done = False     

        self.success = done

        # get success reward
        if self.success:
            reward = self.success_reward
        else:
            reward = 0.0       
   
        return reward, done, self.success

    def get_name(self):
        return "circle_goal"

    def goal_reached(self):
        return self.success 

    def count_goal_object(self):
        return self.goal_object  

    def get_reward_tier(self):
        return self.obj_num - 1 - self.goal_object             
