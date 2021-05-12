import torch
import numpy as np
import os
import gym, ray
from openvrooms.envs.relocate_env import RelocateEnv
from openvrooms.envs.navigate_env import NavigateEnv
from openvrooms.config import *
from openvrooms.envs.vision_env_wrapper import FrameStack, VideoRecorder
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from gym.utils import seeding
import random


# transform OpenRoom env to RLLIB env
class OpenRoomEnvironmentRLLIB(gym.Env): 
    def __init__(self, env_config):
        # make a gym environment
        if env_config["env"] == 'navigate':
            self.env = NavigateEnv(config_file=os.path.join(config_path, env_config["config_file"]), 
            mode=env_config["mode"], device_idx=env_config["device_idx"])

            if env_config["frame_stack"] > 0:
                self.env = FrameStack(self.env, env_config["frame_stack"])
            else:
                print('------------------ No frame stack ------------------')    
        else:
            self.env = RelocateEnv(config_file=os.path.join(config_path, env_config["config_file"]), 
            mode=env_config["mode"], device_idx=env_config["device_idx"])

            if env_config["frame_stack"] > 0:
                self.env = FrameStack(self.env, env_config["frame_stack"])
            else:
                print('------------------ No frame stack ------------------')            

        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(self):
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, float(reward), done, info 

    def seed(self, seed=None):
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]     
        '''
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
            try:
                assert torch is not None
                torch.manual_seed(seed)
                '''
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)
                    torch.cuda.manual_seed_all(seed)
                '''
                #torch.backends.cudnn.enabled = False 
                torch.backends.cudnn.benchmark = False
                torch.backends.cudnn.deterministic = True
            except AssertionError:
                print("Could not seed torch") 


    def get_current_episode_robot_energy_cost(self):
        return self.env.current_episode_robot_energy_cost              

def env_creator(env_config):
    return OpenRoomEnvironmentRLLIB(env_config)  # return an env instance

register_env("openvroom-v0", env_creator)

def test_rllib_env():
    # training
    config = {
        "env": OpenRoomEnvironmentRLLIB,  
        "env_config": {
            "env": "navigate",
            "config_file": 'turtlebot_navigate.yaml',
            "mode": "headless",
            "device_idx": 0
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 2,
        "lr": 1e-4, # try different lrs
        "num_workers": 1,  # parallelism
        "framework": "torch"
    }
    '''
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }
    '''

    
    ray.init()
   
    
    trainer = ppo.PPOTrainer(env=OpenRoomEnvironmentRLLIB, config=config)
    

    while True:
        print(trainer.train())


    #results = tune.run(args.run, config=config, stop=stop)
    
    #print(config["frame_stack"])

if __name__ == "__main__":  
    #test_all_env()   
    test_rllib_env()                           
