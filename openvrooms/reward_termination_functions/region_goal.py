from openvrooms.reward_termination_functions.reward_termination_function_base import BaseRewardTerminationFunction
from gibson2.utils.utils import l2_distance
import numpy as np


class RegionGoal(BaseRewardTerminationFunction):
    """
    Only allow single object
    """

    def __init__(self, config):
        super(RegionGoal, self).__init__(config)
        self.success_reward = float(self.config.get('success_reward', 10.0))

        self.success = False
        

    def reset(self, task, env):
        """
        Compute the initial goal distance after episode reset

        :param task: task instance
        :param env: environment instance
        """
        self.success = False  


    # step
    def get_reward_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if point goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        #pos_distance = task.region_goal_distance()

        # succeed
        cur_xy_pos = task.get_current_xy_position()
        xy_bound = task.region_boundary

        if xy_bound.size == 2:
            if cur_xy_pos[0] > xy_bound[0] and cur_xy_pos[1] > xy_bound[1]:
                done = True
            else:
                done = False 
        elif xy_bound.size == 1:  
            if cur_xy_pos[0] > xy_bound[0]:
                done = True
            else:
                done = False  
        else:
            print("Error: xy_bound has 0 elements")             

        # succeed
        #if pos_distance <= 0:
        #    done = True
        #else:
        #    done = False     

        self.success = done

        # get success reward
        if self.success:
            reward = self.success_reward
        else:
            reward = 0.0       
   
        return reward, done, self.success

    def get_name(self):
        return "region_goal"

    def goal_reached(self):
        return self.success 
             
