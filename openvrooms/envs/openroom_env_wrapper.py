import torch
import numpy as np
import os
import gym, ray
from all.core.state import State
from all.environments.gym import GymEnvironment
from openvrooms.envs.relocate_env import RelocateEnv
from openvrooms.envs.navigate_env import NavigateEnv
from openvrooms.config import *
from openvrooms.envs.vision_env_wrapper import FrameStack, VideoRecorder
from ray.rllib.agents import ppo
from ray.tune.registry import register_env
from gym.utils import seeding


# transform OpenRoom env to gym and ALL env
class OpenRoomEnvironmentALL(GymEnvironment):
    def __init__(self, gym_id, config_file,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,  # pass to simulator
        device=torch.device('cuda:0'), # device used by this environment
        render_to_tensor=False,
        automatic_reset=False,
        save_path=None,
        save_format='gif',
        frame_stack=None):

        # save parameters specific to this class
        self._config_file = config_file
        self._mode = mode
        self._action_timestep = action_timestep
        self._physics_timestep = physics_timestep
        self._device_idx = device_idx
        self._render_to_tensor = render_to_tensor
        self._automatic_reset = automatic_reset
        self._save_path = save_path
        self._save_format = save_format
        self._gym_id = gym_id
        self._frame_stack = frame_stack

        # gym env specific attributes
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}

        # make a gym environment
        env = gym.make(id=gym_id, config_file=config_file, mode=mode, action_timestep=action_timestep, physics_timestep=physics_timestep, device_idx=device_idx,
        render_to_tensor=render_to_tensor, automatic_reset=automatic_reset)

        
        # vision wrapper
        if self._frame_stack:
            env = FrameStack(env, self._frame_stack)

        # video recorder wrapper
        if self._save_path:
            env = VideoRecorder(env, dir_name=self._save_path, file_format=self._save_format, width=env.image_width, height=env.image_width)
        
        # initialize GymEnvironment: self._env = env
        super().__init__(env=env, device=device) 
       
        print("-----------------------------------")
        print("Observation (state) space: ")
        print(self.observation_space)
        #print(self.state_space.shape[0])
        print("-----------------------------------")

    def duplicate(self, n):
        #print("-------duplicate-----")

        return [OpenRoomEnvironment(gym_id=self._gym_id, config_file=self._config_file, mode=self._mode, action_timestep=self._action_timestep,
        physics_timestep=self._physics_timestep,
        device_idx=self._device_idx,  
        device=self._device, 
        render_to_tensor=self._render_to_tensor,
        automatic_reset=self._automatic_reset, save_path=self._save_path) 
        for _ in range(n)] 

    # to use default mode in RelocateEnv instead of gym
    def render(self, mode='rgb'):
        return self._env.render(mode=mode)  

    # for video recorder wrapper
    def start_video_recorder(self, clear=False):
        if self._save_path:
            self._env.start_video_recorder(clear=clear)

    # for video recorder wrapper        
    def stop_video_recorder(self):
        if self._save_path:
            self._env.stop_video_recorder()

    # for video recorder wrapper        
    # gif or mp4
    def save(self):
        if self._save_path:
            self._env.save()              

    @property    
    def observation(self):
        if self._state == None:
            return None

        return self.state['observation']

    # return numpy array on cpus
    @property    
    def observation_np(self):
        if self._state == None:
            return None

        return self.state['observation'].cpu().squeeze().numpy()  

    @property    
    def reward(self):
        if self._state == None:
            return None

        return self.state['reward']

    @property    
    def done(self):
        if self._state == None:
            return False
    
        return self.state['done']   

    # part of info
    # keys in info: 'success', 'episode_length', 'non_interactive_collision_step', 'interactive_collision_step'
    @property    
    def success(self):
        if self._state == None:
            return False

        return self.state['success']     
    


def test_all_env():
    print("Start!") 

    #env = gym.make("openvrooms-v0", config_file=os.path.join(config_path,'turtlebot_relocate.yaml'), mode="headless", action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0) 
    save_path = "/home/meng/openvrooms/learning/runs"
    env = OpenRoomEnvironmentALL(gym_id="openrelocate-v0", config_file=os.path.join(config_path,'turtlebot_relocate.yaml'), mode="headless", action_timestep=1.0 / 10.0, physics_timestep=1.0 / 40.0, frame_stack=4, save_path=save_path, save_format='mp4')
    #print(env.state_space)
    #print(env.observation_space)
    #print(env.action_space)
    
    '''
    #env.start_video_recorder()
    i = 0
    env.reset()
    while i < 100:
        action = env.action_space.sample()
        env.step(torch.from_numpy(np.array([action])))
        #frame = env.render()
        #print(frame.shape)
        
        i += 1

        print("step: %d, action: %s, state: %s, reward: %d, done: %d, success: %s"%(i, action, env.observation_np.shape, env.reward, env.done, env.success))
        print("--------------------------------------------------------")
        
    env.close()
    print("Done!")
    '''
    #env.save()

# transform OpenRoom env to RLLIB env
class OpenRoomEnvironmentRLLIB(gym.Env): 
    def __init__(self, env_config):
        # make a gym environment
        if env_config["env"] == 'navigate':
            self.env = NavigateEnv(config_file=os.path.join(config_path, env_config["config_file"]), 
            mode=env_config["mode"], device_idx=env_config["device_idx"])

            if env_config["frame_stack"] is not None:
                self.env = FrameStack(self.env, env_config["frame_stack"])
            else:
                print('------------------ No frame stack ------------------')    
        else:
            self.env = RelocateEnv(config_file=os.path.join(config_path, env_config["config_file"]), 
            mode=env_config["mode"], device_idx=env_config["device_idx"])

            if env_config["frame_stack"] is not None:
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
        self.np_random, seed = seeding.np_random(seed)
        return [seed]     

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

if __name__ == "__main__":  
    #test_all_env()   
    test_rllib_env()                           
