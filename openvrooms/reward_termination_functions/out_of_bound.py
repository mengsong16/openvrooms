from openvrooms.reward_termination_functions.reward_termination_function_base import BaseRewardTerminationFunction
import numpy as np

class OutOfBound(BaseRewardTerminationFunction):
    """
    Episode terminates if the robot goes outside the valid region
    """

    def __init__(self, config, env):
        super(OutOfBound, self).__init__(config)
        self.safty_thresh = float(self.config.get('body_width', 0.36)) / 2.0 + 0.1
        self.x_bound = np.array(env.scene.x_range)  
        self.y_bound = np.array(env.scene.y_range)

        self.x_bound[0] += self.safty_thresh
        self.x_bound[1] -= self.safty_thresh
        self.y_bound[0] += self.safty_thresh
        self.y_bound[1] -= self.safty_thresh

        '''
        self.x_bound[0] = -1
        self.x_bound[1] = 1
        self.y_bound[0] = -1
        self.y_bound[1] = 1
        '''

        self.reward = self.config.get('out_of_bound_reward', -10.0)

    def get_reward_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if the robot goes outside the valid region
        """
        
        robot_x, robot_y, _ = env.robots[0].get_position()

        #print(robot_x)
        #print(robot_y)

        # Out of valid region
        if robot_x <= self.x_bound[0] or robot_x >= self.x_bound[1] or robot_y <= self.y_bound[0] or robot_y >= self.y_bound[1]:
            reward = self.reward
            done = True
        else:
            reward = 0
            done = False 

        success = False   
        
        return reward, done, success

    def get_name(self):
        return "out_of_bound"        
