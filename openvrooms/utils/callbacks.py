"""Example of using RLlib's debug callbacks.
Here we use callbacks to track the energy as a custom metric.
"""

from typing import Dict
import argparse
import numpy as np
import os

import ray
from ray import tune
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch


class CustomTrainingMetrics(DefaultCallbacks):
    '''
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode, env_index: int, **kwargs):
        # print("episode {} (env-idx={}) started.".format(
        #     episode.episode_id, env_index))

        ## step energy
        #episode.user_data["current_step_robot_energy_cost"] = []
        #episode.user_data["current_step_pushing_energy_cost"] = []
        #episode.hist_data["current_step_robot_energy_cost"] = []
        #episode.hist_data["current_step_pushing_energy_cost"] = []

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        episode: MultiAgentEpisode, env_index: int, **kwargs):
        
        env_list = base_env.get_unwrapped()
        assert len(env_list) == 1, f"Number of unwrapped base_env is not 1: {len(env_list)}!"
        env = env_list[0].env
        ## store step energy
        episode.user_data["current_step_robot_energy_cost"].append(env.current_step_robot_energy_cost)
        episode.user_data["current_step_pushing_energy_cost"].append(env.current_step_pushing_energy_cost)
    '''

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        env_list = base_env.get_unwrapped()
        assert len(env_list) == 1, f"Number of unwrapped base_env is not 1: {len(env_list)}!"
        env = env_list[0].env
        '''
        ## episode energy
        current_episode_robot_energy_cost = env.current_episode_robot_energy_cost
        current_episode_pushing_energy_cost = env.current_episode_pushing_energy_cost
        # print("episode {} (env-idx={}) ended with length {} and pole "
        #       "robot energy {} and pushing energy {}".format(episode.episode_id, env_index, episode.length,
        #                          current_episode_robot_energy_cost, current_episode_pushing_energy_cost))
        episode.custom_metrics["current_episode_robot_energy_cost"] = current_episode_robot_energy_cost
        episode.custom_metrics["current_episode_pushing_energy_cost"] = current_episode_pushing_energy_cost

        ## step energy
        episode.hist_data["current_step_robot_energy_cost"] = episode.user_data["current_step_robot_energy_cost"]
        episode.hist_data["current_step_pushing_energy_cost"] = episode.user_data["current_step_pushing_energy_cost"]
        '''
        episode.custom_metrics["episode_robot_energy"] = env.current_episode_robot_energy_cost
        episode.custom_metrics["episode_pushing_energy"] = env.current_episode_pushing_energy_translation
