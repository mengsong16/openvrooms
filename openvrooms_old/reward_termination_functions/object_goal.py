from openvrooms.reward_termination_functions.reward_termination_function_base import BaseRewardTerminationFunction
from gibson2.utils.utils import l2_distance
import numpy as np


class ObjectGoal(BaseRewardTerminationFunction):
    """
    ObjectGoal used for object relocation tasks
    Episode terminates if point goal is reached
    """

    def __init__(self, config):
        super(ObjectGoal, self).__init__(config)
        self.dist_tol = float(self.config.get('dist_tol', 0.1))
        self.angle_tol = float(self.config.get('angle_tol', 0.2))
        self.success_reward = float(self.config.get('success_reward', 10.0))

        self.use_goal_dist_reward = self.config.get('use_goal_dist_reward', True)
        self.rot_dist_reward_weight = float(self.config.get('rot_dist_reward_weight', 0.2))
        self.goal_dist_reward_weight = float(self.config.get('goal_dist_reward_weight', 1.0))

        self.success = False
        self.goal_object = 0
        self.obj_num = None

    def reset(self, task, env):
        """
        Compute the initial goal distance after episode reset

        :param task: task instance
        :param env: environment instance
        """
        if self.use_goal_dist_reward:
            self.goal_dist = self.get_goal_dist(task)

        self.success = False
        self.goal_object = 0 
        self.obj_num = task.obj_num    

    def get_goal_dist(self, task):
        pos_distances, rot_distances = task.goal_distance()
        return self.compute_weighted_distance(pos_distances, rot_distances)

    def compute_weighted_distance(self, pos_distances, rot_distances):    
        weighted_distance = np.mean(pos_distances) + self.rot_dist_reward_weight * np.mean(rot_distances)
        return weighted_distance

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

        '''
        # check success / done
        done = True
        for i in list(range(task.obj_num)):
            #if pos_distances[i] > self.dist_tol or rot_distances[i] > self.angle_tol:
            if pos_distances[i] > self.dist_tol:
                done = False
                break
        '''
        # count how many objects are in the goal position
        self.goal_object = 0

        # check success / done
        for i in list(range(task.obj_num)):
            #if pos_distances[i] > self.dist_tol or rot_distances[i] > self.angle_tol:
            if pos_distances[i] <= self.dist_tol:
                self.goal_object += 1
        
        # succeed
        if self.goal_object == task.obj_num:
            done = True
        else:
            done = False     

        self.success = done

        # get success reward
        if self.success:
            reward = self.success_reward
        else:
            reward = 0.0    
        
        # get goal reward
        if self.use_goal_dist_reward:
            new_goal_dist = self.compute_weighted_distance(pos_distances, rot_distances)
            goal_dist_reward = self.goal_dist_reward_weight * (self.goal_dist - new_goal_dist)
            self.goal_dist = new_goal_dist

            reward += goal_dist_reward
   

        return reward, done, self.success

    def get_name(self):
        return "object_goal"

    def goal_reached(self):
        return self.success 

    def count_goal_object(self):
        return self.goal_object  

    def get_reward_tier(self):
        return self.obj_num - 1 - self.goal_object             
