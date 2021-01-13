from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition
import numpy as np

class OutOfBound(BaseTerminationCondition):
    """
    Episode terminates if the robot goes outside the valid region
    """

    def __init__(self, config, env):
        super(OutOfBound, self).__init__(config)
        self.safty_thresh = float(self.config.get('dist_tol', 0.36)) / 2.0 + 0.1
        self.x_bound = np.array(env.scene.x_range)  
        self.y_bound = np.array(env.scene.y_range)

        self.x_bound[0] += self.safty_thresh
        self.x_bound[1] -= self.safty_thresh
        self.y_bound[0] += self.safty_thresh
        self.y_bound[1] -= self.safty_thresh

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if the robot goes outside the valid region
        """
        
        robot_x, robot_y, _ = env.robots[0].get_position()

        print(robot_x)
        print(robot_y)

        # Out of valid region
        if robot_x <= self.x_bound[0] or robot_x >= self.x_bound[1] or robot_y <= self.y_bound[0] or robot_y >= self.y_bound[1]:
            done = True
        else:
            done = False 

        success = False   
        
        return done, success
