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

        #self.get_wheels()  
        self.separate_wheel_non_wheel()

        # disable non wheel joints  
        #self.disable_non_wheel()
        return ids

    '''
    def get_wheels(self):
        self.wheel_joints = []
        wheel_names = ['l_wheel_joint', 'r_wheel_joint']
        for name in wheel_names:
            for j in self.ordered_joints:
                if j.joint_name == name:
                    self.wheel_joints.append(j)
                    break
    '''                
    def separate_wheel_non_wheel(self):
        self.wheel_joints = []
        self.non_wheel_joints = []
        wheel_names = ['l_wheel_joint', 'r_wheel_joint']
        for j in self.ordered_joints:
            if j.joint_name in wheel_names:
                self.wheel_joints.append(j)
            else:
                self.non_wheel_joints.append(j)   

        print("----------------------------------------------")
        print('Wheeled joints: %d'%(len(self.wheel_joints))) 
        print('Non-wheeled joints: %d'%(len(self.non_wheel_joints))) 
        print("----------------------------------------------")        
    
    def disable_non_wheel(self):
        print("----------------------------------------------")
        for j in self.non_wheel_joints:    
            j.disable_motor()
            print("Disabled %s"%(j.joint_name))

        print("----------------------------------------------")    

           
    def print_joint_info(self): 
        print("%d Joints"%(len(self.ordered_joints)))
        for j in self.ordered_joints:
            _, vel, trq = j.get_state()
            _, rvel, rtrq = j.get_relative_state()
            print("----------------------------------------------")
            print("%s: "%(j.joint_name))
            print("max_velocity: %f, current_absolute_velocity: %f, current_relative_velocity: %f"%(j.max_velocity, vel, rvel))
            print("max_torque: %f, current_absolute_torque: %f, current_relative_torque:%f"%(j.max_torque, trq, rtrq))
            print("max_energy_cost: %f, current_absolute_energy_cost: %f, current_relative_energy_cost:%f"%(np.abs(j.max_velocity*j.max_torque), np.abs(vel*trq), np.abs(rvel*rtrq)))
            print("----------------------------------------------")

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

    # get raw and max joint velocity and torque
    # range: [-1,1] (try to, may overflow)
    # only consider wheel joints
    # setted_wheel_velocity is in [0,1]
    def get_joint_state(self, discrete_action_space, setted_wheel_velocity):
        # [n,3]: n joints

        # get raw joint velocity and torque
        joint_info = np.array([j.get_state() for j in self.wheel_joints]).astype(np.float32)    
        #joint_position = joint_info[:,0]
        joint_raw_velocity = joint_info[:,1]
        joint_raw_torque = joint_info[:,2]

        joint_max_velocity = np.array([j.max_velocity for j in self.wheel_joints]).astype(np.float32)
        if discrete_action_space:
           joint_max_velocity *= abs(float(setted_wheel_velocity))
             
        # Note that since we are using velocity control, the real torque could be larger than mox torque     
        joint_max_torque = np.array([j.max_torque for j in self.wheel_joints]).astype(np.float32) 
        # print("---------------------------")
        # print("Joint raw velocity: %s"%(joint_raw_velocity))
        # print("Joint raw torque: %s"%(joint_raw_torque))
        # print("Joint max velocity: %s"%(joint_max_velocity))
        # print("Joint max torque: %s"%(joint_max_torque))
        # print("---------------------------")

        return joint_raw_velocity, joint_raw_torque, joint_max_velocity, joint_max_torque

    def get_energy(self, normalized=True, discrete_action_space=False, setted_wheel_velocity=1.0):
        joint_raw_velocity, joint_raw_torque, joint_max_velocity, joint_max_torque = self.get_joint_state(discrete_action_space, setted_wheel_velocity)
        #print("joint velocity: %s"%(joint_raw_velocity))
        #print("joint torque: %s"%(joint_raw_torque))
        raw_energy = np.abs(np.dot(joint_raw_velocity, joint_raw_torque))

        if normalized:
            max_energy = np.abs(np.dot(joint_max_velocity, joint_max_torque))
            normalized_energy = float(raw_energy) / float(max_energy)
            #print("---------------------------")
            #print(raw_energy)
            #print(max_energy)
            #print(normalized_energy)
            #print("---------------------------")
            return normalized_energy
        #print("Energy cost: %f"%energy)
        else:
            # print("---------------------------")
            # print(raw_energy)
            # print("---------------------------")
            return raw_energy
          
