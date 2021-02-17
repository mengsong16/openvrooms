import torch
import numpy as np
import os
import gym, ray

import random

from openvrooms.envs.openroom_env_wrapper import OpenRoomEnvironmentRLLIB
from openvrooms.config import *


from ray.rllib.agents import ppo, dqn

from ray import tune
from ray.tune.logger import pretty_print

# def set_seed_everywhere(seed):
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)


#-------------------- config -------------------
#env_option =  'navigate' 
env_option = 'relocate'


dqn_train_config = dqn.DEFAULT_CONFIG.copy()
dqn_train_config = {
        "env": OpenRoomEnvironmentRLLIB,  
        "env_config": {
            "env": env_option,
            "config_file": 'turtlebot_%s.yaml'%(env_option),
            "mode": "headless",
            "device_idx": 0, # renderer use gpu 0
            "frame_stack": None
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 2,
        "lr": 1e-4, # try different lrs
        "framework": "torch",
        "seed": 1,
        "train_batch_size": 64,
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 10000,  # Timesteps over which to anneal epsilon.

            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        }
    }

ppo_train_config = ppo.DEFAULT_CONFIG.copy()
ppo_train_config = {
        #"env": "Breakout-v0",
        "env": OpenRoomEnvironmentRLLIB,  
        "env_config": {
           "env": env_option,
           "config_file": 'turtlebot_%s.yaml'%(env_option),
           "mode": "headless",
           "device_idx": 0, # renderer use gpu 0
           "frame_stack": 4
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 2,
        "num_workers": 10,
        "lr": 1e-4, # try different lrs
        "framework": "torch",
        "seed": 1,
        "train_batch_size": 512,
        #"model": {
        #"dim": 128, 
        #"conv_filters":[[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]],
        #"num_framestacks": 4
        #}
}


stop = {
        #"timesteps_total": 350000,
        "episode_reward_mean": 520,
    }

def print_model():
    ray.init()

    agent = ppo.PPOTrainer(config=ppo_train_config)
    policy = agent.get_policy()

    print(policy.model)

def train_ppo():
    #torch.backends.cudnn.enabled = False 
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

    ray.init()
    #set_seed_everywhere(seed=1)

    results = tune.run("PPO", config=ppo_train_config, stop=stop, checkpoint_at_end=True)

def train_dqn():
    ray.init()
    
    '''
    trainer = dqn.DQNTrainer(config=dqn_config)

    for i in range(1000):
       # Perform one iteration of training the policy with PPO
       result = trainer.train()
       #print(pretty_print(result))
       print("Iter %d: min: %f, mean: %f, max: %f, len: %d"%(i, result["episode_reward_min"], result["episode_reward_mean"], 
            result["episode_reward_max"], result["episode_len_mean"]))

       if i % 100 == 0:
           checkpoint = trainer.save()
           print("checkpoint saved at", checkpoint)
    '''

    results = tune.run("DQN", config=dqn_train_config, stop=stop, checkpoint_at_end=True)

if __name__ == "__main__":    
    #train_dqn()
    train_ppo() 
    #print_model()