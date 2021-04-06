from openvrooms.reward_termination_functions.reward_termination_function_base import BaseRewardTerminationFunction
import numpy as np

class PosNegCollision(BaseRewardTerminationFunction):
    """
    Collision
    Penalize collision with non-interactive objects, encourate collision with interactive objects.
    """

    def __init__(self, config):
        super(PosNegCollision, self).__init__(config)

        self.collision_penalty = float(self.config.get('collision_penalty', -1))
        self.collision_reward = float(self.config.get('collision_reward', 0.1))
        self.max_collisions_allowed = int(self.config.get('max_collisions_allowed', 500))

        self.has_interactive_collision = False
        self.has_non_interactive_collision = False


    def get_reward_termination(self, task, env):
        """
        Reward is self.collision_reward_weight if there is collision
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        self.has_non_interactive_collision = (len(env.non_interactive_collision_links) > 0)
        self.has_interactive_collision = (len(env.interactive_collision_links) > 0)

        '''
        print("***********************************")
        if has_interactive_collision:
            print("interactive collision!")
            print(env.interactive_collision_links)

        if has_non_interactive_collision:
            print("non-interactive collision!") 
            print(env.non_interactive_collision_links)
        print("***********************************")    
        '''   
        
        reward = float(self.has_non_interactive_collision) * self.collision_penalty + float(self.has_interactive_collision) * self.collision_reward

        # collide with non-interactive objects reach maximum times
        done = env.non_interactive_collision_step > self.max_collisions_allowed
        success = False

        return reward, done, success

    def get_name(self):
        return "negative_and_positive_collision"  

    def has_positive_collision(self):
        return self.has_interactive_collision

    def has_negative_collision(self):
        return self.has_non_interactive_collision          

class NegCollision(BaseRewardTerminationFunction):
    """
    Collision
    Penalize collision with obstacles.
    """

    def __init__(self, config):
        super(NegCollision, self).__init__(config)

        self.collision_penalty = self.config.get('collision_penalty', -1)
        self.max_collisions_allowed = self.config.get('max_collisions_allowed', 500)

        self.has_collision = False

    def get_reward_termination(self, task, env):
        """
        Reward is self.collision_reward_weight if there is collision
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        self.has_collision = (len(env.collision_links) > 0) 
        
        '''
        print("***********************************")
        if has_interactive_collision:
            print("interactive collision!")
            print(env.interactive_collision_links)

        if has_non_interactive_collision:
            print("non-interactive collision!") 
            print(env.non_interactive_collision_links)
        print("***********************************")    
        '''   
        
        reward = float(self.has_collision) * self.collision_penalty

        # collide with non-interactive objects reach maximum times
        done = env.collision_step > self.max_collisions_allowed
        success = False

        return reward, done, success

    def get_name(self):
        return "negative_collision" 

    def has_negative_collision(self):
        return self.has_collision            