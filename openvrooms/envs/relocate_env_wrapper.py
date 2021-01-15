import torch
import numpy as np
import os
import gym
from all.core.state import State
from all.environments.gym import GymEnvironment
from openvrooms.envs.relocate_env import RelocateEnv
from openvrooms.config import *


class OpenRelocateEnvironment(GymEnvironment):
    def __init__(self, gym_id, config_file,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,  # pass to simulator
        device=torch.device('cuda:0'), # device used by this environment
        render_to_tensor=False,
        automatic_reset=False,
        save_path=None):

        # save parameters specific to this class
        self._config_file = config_file
        self._mode = mode
        self._action_timestep = action_timestep
        self._physics_timestep = physics_timestep
        self._device_idx = device_idx
        self._render_to_tensor = render_to_tensor
        self._automatic_reset = automatic_reset
        self._save_path = save_path
        self._gym_id = gym_id


        # make a gym environment
        env = gym.make(id=gym_id, config_file=config_file, mode=mode, action_timestep=action_timestep, physics_timestep=physics_timestep, device_idx=device_idx,
        render_to_tensor=render_to_tensor, automatic_reset=automatic_reset)
        
        # monitor wrapper
        if self._save_path:
            env = gym.wrappers.Monitor(env, self._save_path, force=True)
       
        # initialize
        super().__init__(env=env, device=device) 
       


    def duplicate(self, n):
        print("-------duplicate-----")

        return [OpenRelocateEnvironment(gym_id=self._gym_id, config_file=self._config_file, mode=self._mode, action_timestep=self._action_timestep,
        physics_timestep=self._physics_timestep,
        device_idx=self._device_idx,  
        device=self._device, 
        render_to_tensor=self._render_to_tensor,
        automatic_reset=self._automatic_reset, save_path=self._save_path) 
        for _ in range(n)] 

    @property    
    def observation(self):
        return self.state['observation']

    # return numpy array on cpus
    @property    
    def observation_np(self):
        return self.state['observation'][0].cpu().squeeze().numpy()  

    @property    
    def reward(self):
        return self.state['reward']

    @property    
    def done(self):
        return self.state['done']   

    # part of info
    # keys in info: 'success', 'episode_length', 'non_interactive_collision_step', 'interactive_collision_step'
    @property    
    def success(self):
        return self.state['success']     
    


def test_env():
    print("Start!") 

    #env = gym.make("openvrooms-v0", config_file=os.path.join(config_path,'turtlebot_relocate.yaml'), mode="headless", action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0) 

    env = OpenRelocateEnvironment(gym_id="openrelocate-v0", config_file=os.path.join(config_path,'turtlebot_relocate.yaml'), mode="headless", action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0)
    #print(env._name)
    #print(env._env)  
    print(env.name)
    print(env.duplicate(1)) 
    print(env.name)

    #print(env.action_space)
    #print(env.state_space)
    '''
    i = 0
    env.reset()
    while i < 2000:
        action = env.action_space.sample()
        env.step(torch.from_numpy(np.array([action])))
        
        i += 1

        print("step: %d, action: %s, state: %s, reward: %d, done: %d, success: %s"%(i, action, env.observation, env.reward, env.done, env.success))
        print("--------------------------------------------------------")
        
    env.close()
    print("Done!")
    '''

    

if __name__ == "__main__":  
    test_env()                              
