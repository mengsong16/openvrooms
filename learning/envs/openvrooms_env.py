import torch
import numpy as np
import gym
from all.core.state import State
from all.environments.gym import GymEnvironment
import openvrooms
from openvrooms.envs.relocate_env import RelocateEnv

class OpenvroomsEnvironment(GymEnvironment):
    def __init__(self, name, save_path=None, *args, **kwargs):
        # need these for duplication
        self._args = args
        self._kwargs = kwargs
        # make a gym environment
        env = gym.make(name, start=start, goal=goal, goal_conditioned=goal_conditioned, random_start=random_start, wall_illegal=wall_illegal)
        # monitor wrapper
        if save_path:
            env = gym.wrappers.Monitor(env, save_path, force=True)
        # initialize
        super().__init__(env, *args, **kwargs) # self._done is True
        self._env = env 
        

    
        #print('-----------------------')
        #print(env._max_episode_steps)  #2000
        #print('-----------------------')
        
        
        
    # self._state is State
    # self._desired_goal and self._achieved_goal are raw data (numpy array) 
    # restart an episode   
    def reset(self):
        # Note that super().reset() will call MazeEnv's _make_state() instead of GymEnv's
        #init_state = super().reset()
        super()._lazy_init()
        #print(self._env.goal_conditioned)
        #print(self._env.random_start)
        raw_state = self._env.reset()
        # 0 means not done, it is not reward
        # self._state is State
        self._state = self._make_state(raw=raw_state, done=0, info=action_mask)
        # initialize _reward=-1, although the reward is undefined when reset() is called
        self._reward = -1
        #self._reward = 1
        self._done = False


    @property
    def name(self):
        return self._name

    def duplicate(self, n):
        print("-------duplicate-----")
        return [OpenvroomsEnvironment(self._name, self._save_path, *self._args, **self._kwargs) for _ in range(n)]


    # wrap done and info into state
    # step() will call this function instead super()._make_state()?
    # ensure that raw is a numpy with a specific type consistant with goal state  
    # info is a numpy array of action mask
    def _make_state(self, raw, done, info=None):
        #print("Maze")
        #Convert numpy array into State
        return State(
            torch.from_numpy(np.array(
                    raw,
                    dtype=self.state_space.dtype
                )).unsqueeze(0).to(self._device),
            self._done_mask if done else self._not_done_mask,
            [torch.from_numpy(info).to(self._device)] if info is not None else []
        )

    def step(self, action):
        # will call this _make_state, not super's _make_state
        super().step(action)
        #if self._goal_conditioned:
        self._achieved_goal = self.raw_state

        return self._state, self._reward

    # keep action the same, still a Tensor
    # called by heritated step()     
    def _convert(self, action):
        if isinstance(self.action_space, gym.spaces.Discrete):
            return action
        raise TypeError("Unknown action space type")  
         

    @property    
    def raw_state(self):
        return self._state._raw.cpu().squeeze().numpy()       
    
        
if __name__ == "__main__":  
    print("Start!") 

    env = gym.make("openvrooms-v0") 
    print("Succeed!")                          
