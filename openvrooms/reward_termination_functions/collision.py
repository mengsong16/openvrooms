from openvrooms.reward_termination_functions.reward_termination_function_base import BaseRewardTerminationFunction
import numpy as np

class PosNegCollision(BaseRewardTerminationFunction):
    """
    Collision
    Penalize collision with non-interactive objects, encourate collision with interactive objects.
    """

    def __init__(self, config):
        super(PosNegCollision, self).__init__(config)

        self.collision_penalty = self.config.get('collision_penalty', -1)
        self.collision_reward = self.config.get('collision_reward', 0.1)
        self.max_collisions_allowed = self.config.get('max_collisions_allowed', 500)

    def get_reward_termination(self, task, env):
        """
        Reward is self.collision_reward_weight if there is collision
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        has_non_interactive_collision = float(len(env.non_interactive_collision_links) > 0)
        has_interactive_collision = float(len(env.interactive_collision_links) > 0)

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
        
        reward = has_non_interactive_collision * self.collision_penalty + has_interactive_collision * self.collision_reward

        # collide with non-interactive objects reach maximum times
        done = env.non_interactive_collision_step > self.max_collisions_allowed
        success = False

        return reward, done, success

    def get_name(self):
        return "collision"    

class NegCollision(BaseRewardTerminationFunction):
    """
    Collision
    Penalize collision with obstacles.
    """

    def __init__(self, config):
        super(NegCollision, self).__init__(config)

        self.collision_penalty = self.config.get('collision_penalty', -1)
        self.max_collisions_allowed = self.config.get('max_collisions_allowed', 500)

    def get_reward_termination(self, task, env):
        """
        Reward is self.collision_reward_weight if there is collision
        in the last timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        collision = float(len(env.collision_links) > 0)
        
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
        
        reward = collision * self.collision_penalty

        # collide with non-interactive objects reach maximum times
        done = env.collision_step > self.max_collisions_allowed
        success = False

        return reward, done, success

    def get_name(self):
        return "collision"         