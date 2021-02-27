import gym
import numpy as np
import pybullet as p

from gibson2.external.pybullet_tools.utils import joints_from_names, set_joint_positions
from gibson2.robots.robot_locomotor import LocomotorRobot


class Fetch(LocomotorRobot):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    Uses joint velocity control
    """

    def __init__(self, config):
        self.config = config
        self.wheel_velocity = config.get('wheel_velocity', 1.0)
        self.torso_lift_velocity = config.get('torso_lift_velocity', 1.0)
        self.arm_velocity = config.get('arm_velocity', 1.0)
        self.wheel_dim = 2
        self.torso_lift_dim = 1
        self.arm_dim = 7
        self.joint_control = config.get("joint_control", True)

        '''
        LocomotorRobot.__init__(self,
                                "fetch/fetch.urdf",
                                action_dim=self.wheel_dim + self.torso_lift_dim + self.arm_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity",
                                self_collision=True)
        '''
        LocomotorRobot.__init__(self,
                                "fetch/fetch.urdf",
                                action_dim=self.wheel_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity",
                                self_collision=True)                        

    '''
    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_high = np.array([self.wheel_velocity] * self.wheel_dim +
                                    [self.torso_lift_velocity] * self.torso_lift_dim +
                                    [self.arm_velocity] * self.arm_dim)
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
    '''
    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_high = np.array([self.wheel_velocity] * self.wheel_dim)
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)    

    '''
    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        assert False, "Fetch does not support discrete actions"
    '''
    # joint level or agent level
    
    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        if self.joint_control:
            self.action_list = [[self.wheel_velocity, self.wheel_velocity], 
                            [self.wheel_velocity * 0.5, -self.wheel_velocity * 0.5],
                            [-self.wheel_velocity * 0.5, self.wheel_velocity * 0.5], [0, 0]]
        else:
            self.action_list = list(range(4))

        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action() 

    '''
    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        if self.joint_control:
            self.action_list = [[self.wheel_velocity, self.wheel_velocity], [-self.wheel_velocity, -self.wheel_velocity],
                            [self.wheel_velocity * 0.5, -self.wheel_velocity * 0.5],
                            [-self.wheel_velocity * 0.5, self.wheel_velocity * 0.5], [0, 0]]
        else:
            self.action_list = list(range(5))

        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()
    '''
    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # forward
            (ord('s'),): 1,  # backward
            (ord('d'),): 2,  # turn right
            (ord('a'),): 3,  # turn left
            (): 4  # stay still
        }                

    def robot_specific_reset(self):
        """
        Fetch robot specific reset.
        Reset the torso lift joint and tuck the arm towards the body
        """
        super(Fetch, self).robot_specific_reset()

        # roll the arm to its body
        robot_id = self.robot_ids[0]
        arm_joints = joints_from_names(robot_id,
                                       [
                                           'torso_lift_joint',
                                           'shoulder_pan_joint',
                                           'shoulder_lift_joint',
                                           'upperarm_roll_joint',
                                           'elbow_flex_joint',
                                           'forearm_roll_joint',
                                           'wrist_flex_joint',
                                           'wrist_roll_joint'
                                       ])

        rest_position = (0.02, np.pi / 2.0 - 0.4, np.pi / 2.0 -
                         0.1, -0.4, np.pi / 2.0 + 0.1, 0.0, np.pi / 2.0, 0.0)
        # might be a better pose to initiate manipulation
        # rest_position = (0.30322468280792236, -1.414019864768982,
        #                  1.5178184935241699, 0.8189625336474915,
        #                  2.200358942909668, 2.9631312579803466,
        #                  -1.2862852996643066, 0.0008453550418615341)

        set_joint_positions(robot_id, arm_joints, rest_position)

    def get_end_effector_position(self):
        """
        Get end-effector position
        """
        return self.parts['gripper_link'].get_position()

    def end_effector_part_index(self):
        """
        Get end-effector link id
        """
        return self.parts['gripper_link'].body_part_index

    def load(self):
        """
        Load the robot into pybullet. Filter out unnecessary self collision
        due to modeling imperfection in the URDF
        """
        ids = super(Fetch, self).load()
        robot_id = self.robot_ids[0]

        disable_collision_names = [
            ['torso_lift_joint', 'shoulder_lift_joint'],
            ['torso_lift_joint', 'torso_fixed_joint'],
            ['caster_wheel_joint', 'estop_joint'],
            ['caster_wheel_joint', 'laser_joint'],
            ['caster_wheel_joint', 'torso_fixed_joint'],
            ['caster_wheel_joint', 'l_wheel_joint'],
            ['caster_wheel_joint', 'r_wheel_joint'],
        ]
        for names in disable_collision_names:
            link_a, link_b = joints_from_names(robot_id, names)
            p.setCollisionFilterPair(robot_id, robot_id, link_a, link_b, 0)

        self.get_wheels()    

        return ids

    def get_wheels(self):
        self.wheel_joints = []
        wheel_names = ['l_wheel_joint', 'r_wheel_joint']
        for name in wheel_names:
            for j in self.ordered_joints:
                if j.joint_name == name:
                    self.wheel_joints.append(j)
                    break
           

    def apply_robot_action(self, action):
        """
        Apply robot action.
        Support joint torque, velocity, position control and
        differentiable drive / twist command control

        :param action: robot action
        """
        if self.control == 'velocity':
            for n, j in enumerate(self.wheel_joints):
                #print(j.joint_name)
                j.set_motor_velocity(
                    self.velocity_coef * j.max_velocity * float(np.clip(action[n], -1, +1)))
              
            #print('---------------------------------------')      
        else:
            raise Exception('unknown control type: {}'.format(self.control))  

    def get_mass(self):
        return self.robot_mass

    # get normalized joint velocity and torque
    # range: [-1,1]
    def get_joint_info(self, normalized):
        # [n,3]: n joints
        if normalized:
            joint_info = np.array([j.get_relative_state() for j in self.ordered_joints]).astype(np.float32)
        else:
            joint_info = np.array([j.get_state() for j in self.ordered_joints]).astype(np.float32)    

        joint_position = joint_info[:,0]
        joint_velocity = joint_info[:,1]
        joint_torque = joint_info[:,2]

    
        #print("---------------------------")
        #print("Joint velocity: %s"%(joint_velocity))
        #print("Joint torque: %s"%(joint_torque))


        return joint_velocity, joint_torque

    def get_energy(self, normalized=True, discrete_action_space=False, wheel_velocity=1.0):
        joint_velocity, joint_torque = self.get_joint_info(normalized)
        #print("joint velocity: %s"%(joint_velocity))
        #print("joint torque: %s"%(joint_torque))
        energy = np.abs(joint_velocity * joint_torque).mean()


        if discrete_action_space:
            energy /= abs(float(wheel_velocity))
        #print("Energy cost: %f"%energy)

        return energy
          
