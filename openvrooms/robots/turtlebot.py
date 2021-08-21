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
        self.joint_control = config.get("joint_control", True)

        LocomotorRobot.__init__(self,
                                robot_urdf,
                                action_dim=2,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity")

        if self.is_discrete == True and self.joint_control == False:
            self.move_distance = config.get("move_distance", 0.05)
            self.turn_angle = config.get("turn_angle", 0.03)
            if self.move_distance <= 0 or self.turn_angle <= 0:
                raise Exception("Minimum moving distance and turning angle should be non-negative!")
        '''
        print("---------------------------------")
        print("My turtlebot")
        print("---------------------------------")
        '''

    # joint level
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

    # joint level or agent level
    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        if self.joint_control:
            self.action_list = [[self.velocity, self.velocity], [-self.velocity, -self.velocity],
                            [self.velocity * 0.5, -self.velocity * 0.5],
                            [-self.velocity * 0.5, self.velocity * 0.5], [0, 0]]
        else:
            self.action_list = list(range(5))

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

    def apply_action(self, action):
        """
        Apply policy action
        """
        if self.joint_control:
            real_action = self.policy_action_to_robot_action(action)
            self.apply_robot_action(real_action)
        else:
            if self.is_discrete:
                if action == 0:
                    self.keep_still()
                elif action == 1:    
                    self.move_forward(forward=self.move_distance)
                elif action == 2:    
                    self.move_backward(backward=self.move_distance)
                elif action == 3:
                    self.turn_left(delta=self.turn_angle)
                elif action == 4:
                    self.turn_right(delta=self.turn_angle)
                else:
                    raise Exception("Undefined discrete action: %d"%(action))
            else:
                raise Exception("Not implemented agent level continuous control!")

    # set wheel velocity, range: [-1,1]
    def set_velocity(self, velocity):
        self.velocity = velocity

    def get_mass(self):
        return self.robot_mass
        
    # get normalized joint velocity and torque
    # range: [-1,1] (try to, may overflow)
    # all ordered joints are wheel joints
    def get_joint_state(self, normalized):
        # [n,3]: n joints
        if normalized:
            joint_info = np.array([j.get_relative_state() for j in self.ordered_joints]).astype(np.float32)
        else:
            joint_info = np.array([j.get_state() for j in self.ordered_joints]).astype(np.float32)    

        #joint_position = joint_info[:,0]
        joint_velocity = joint_info[:,1]
        joint_torque = joint_info[:,2]

    
        #print("---------------------------")
        #print("Joint velocity: %s"%(joint_velocity))
        #print("Joint torque: %s"%(joint_torque))


        return joint_velocity, joint_torque

    def get_energy(self, physics_simulation_timestep, normalized=True, discrete_action_space=False, setted_wheel_velocity=1.0):
        joint_raw_velocity, joint_raw_torque, joint_max_velocity, joint_max_torque = self.get_joint_state(discrete_action_space, setted_wheel_velocity)
        #print("joint velocity: %s"%(joint_raw_velocity))
        #print("joint torque: %s"%(joint_raw_torque))
        raw_power = np.abs(np.dot(joint_raw_velocity, joint_raw_torque))

        if normalized:
            max_power = np.abs(np.dot(joint_max_velocity, joint_max_torque))
            normalized_power = float(raw_power) / float(max_power)
            #print("---------------------------")
            #print(raw_energy)
            #print(max_energy)
            #print(normalized_joint_energy)
            #print("---------------------------")
            return normalized_power * physics_simulation_timestep
        #print("Energy cost: %f"%energy)
        else:
            # print("---------------------------")
            # print(raw_energy)
            # print("---------------------------")
            return raw_power * physics_simulation_timestep

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

