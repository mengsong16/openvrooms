import gym
import numpy as np

from gibson2.robots.robot_locomotor import LocomotorRobot


class Turtlebot(LocomotorRobot):
    """
    Turtlebot robot
    Reference: http://wiki.ros.org/Robots/TurtleBot
    Uses joint velocity control
    """

    def __init__(self, config, robot_urdf="turtlebot/turtlebot.urdf"):
        self.config = config
        self.velocity = config.get("velocity", 1.0)
        LocomotorRobot.__init__(self,
                                robot_urdf,
                                action_dim=2,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity")

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = np.full(
            shape=self.action_dim, fill_value=self.velocity)
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        self.action_list = [[self.velocity, self.velocity], [-self.velocity, -self.velocity],
                            [self.velocity * 0.5, -self.velocity * 0.5],
                            [-self.velocity * 0.5, self.velocity * 0.5], [0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # forward
            (ord('s'),): 1,  # backward
            (ord('d'),): 2,  # turn right
            (ord('a'),): 3,  # turn left
            (): 4  # stay still
        }

    # range: [-1,1]
    def set_velocity(self, velocity):
        self.velocity = velocity
        # reset action spaces
        self.set_up_discrete_action_space()
        self.set_up_continuous_action_space()

    # get normalized joint velocity and torque
    # range: [-1,1]
    def get_joint_info(self):
        j = np.array([j.get_joint_relative_state() for j in self.ordered_joints]).astype(np.float32).flatten()
        joint_position = j[0::3]
        joint_velocity = j[1::3]
        joint_torque = j[2::3]

        '''
        print("---------------------------")
        print("Joint velocity: %s"%(joint_velocity))
        print("Joint torque: %s"%(joint_torque))
        print("---------------------------")
        '''

        return joint_velocity, joint_torque     
