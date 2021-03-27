import torch
import numpy as np
import os
import gym, ray

import random

from openvrooms.envs.openroom_env_wrapper import OpenRoomEnvironmentRLLIB
from openvrooms.config import *


from ray.rllib.agents import ppo, dqn, sac

from ray import tune
from ray.tune.logger import pretty_print



#-------------------- config -------------------
#env_option =  'navigate' 
env_option = 'relocate'

robot_option = 'fetch'
#robot_option = 'turtlebot'


#dqn_train_config = dqn.DEFAULT_CONFIG.copy()
dqn_train_config = {
        "env": OpenRoomEnvironmentRLLIB,  
        "env_config": {
            "env": env_option,
            "config_file": '%s_%s.yaml'%(robot_option, env_option),
            "mode": "headless",
            "device_idx": 0, # renderer use gpu 0
            "frame_stack": 0
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1,
        "lr": 1e-4, # try different lrs
        "framework": "torch",
        "seed": 1,
        "train_batch_size": 512, # Size of a batch sampled from replay buffer for training.
        "buffer_size": 50000,
        "exploration_config": {
            # The Exploration class to use.
            "type": "EpsilonGreedy",
            # Config for the Exploration class' constructor:
            "initial_epsilon": 1.0,
            "final_epsilon": 0.02,
            "epsilon_timesteps": 500000,  # Timesteps over which to anneal epsilon.
            #"epsilon_schedule": {"value": 0.1},
            
            # For soft_q, use:
            # "exploration_config" = {
            #   "type": "SoftQ"
            #   "temperature": [float, e.g. 1.0]
            # }
        }
    }

#ppo_train_config = ppo.DEFAULT_CONFIG.copy()
ppo_train_config = {
        #"env": "Breakout-v0",
        "env": OpenRoomEnvironmentRLLIB,  
        "env_config": {
           "env": env_option,
           "config_file": '%s_%s.yaml'%(robot_option, env_option),
           #"config_file": 'fetch_relocate_different_objects.yaml',
           "mode": "headless",
           "device_idx": 0, # renderer use gpu 0
           "frame_stack": 0
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1,
        "num_workers": 10,
        "lr": 1e-4, # try different lrs
        "framework": "torch",
        "seed": 1,
        "train_batch_size": 4000,#8192, #4000,
        "sgd_minibatch_size": 512,
        #"model": {
        #"dim": 128, 
        #"conv_filters":[[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]],
        #"num_framestacks": 4
        #},
        "lambda": 0.98,
        "clip_param": 0.3,
        "entropy_coeff": 0.0,
        "kl_coeff": 0.3,
        "kl_target": 0.01
}

#sac_train_config = sac.DEFAULT_CONFIG.copy()
sac_train_config = {
        #"env": "Breakout-v0",
        "env": OpenRoomEnvironmentRLLIB,  
        "env_config": {
           "env": env_option,
           "config_file": '%s_%s.yaml'%(robot_option, env_option),
           "mode": "headless",
           "device_idx": 0, # renderer use gpu 0
           "frame_stack": 0
        },
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 1,
        "num_workers": 10,
        "lr": 1e-4, # try different lrs
        "framework": "torch",
        "seed": 1,
        "train_batch_size": 512,
        #"model": {
        #"dim": 128, 
        #"conv_filters":[[16, [4, 4], 2], [32, [4, 4], 2], [512, [11, 11], 1]],
        #"num_framestacks": 4
        #},
}


stop = {
        "timesteps_total": 700000 #1000000, #3000000,
        #"episode_reward_mean": 0,
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

    results = tune.run("PPO", config=ppo_train_config, stop=stop, checkpoint_freq=100, checkpoint_at_end=True)



def train_sac():
    #torch.backends.cudnn.enabled = False 
    #torch.backends.cudnn.benchmark = False
    #torch.backends.cudnn.deterministic = True

    ray.init()

    results = tune.run("SAC", config=sac_train_config, stop=stop, checkpoint_freq=100, checkpoint_at_end=True)    

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

    results = tune.run("DQN", config=dqn_train_config, stop=stop, checkpoint_freq=100, checkpoint_at_end=True)

if __name__ == "__main__":    
    #train_dqn()
    train_ppo()
    #train_sac() 
    #print_model()
    