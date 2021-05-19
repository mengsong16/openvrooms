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
from ray.tune.logger import LoggerCallback
from openvrooms.config import *
from shutil import copyfile

class CustomTrainingMetrics(DefaultCallbacks):
	def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
					   policies: Dict[str, Policy], episode: MultiAgentEpisode,
					   env_index: int, **kwargs):
		env_list = base_env.get_unwrapped()
		assert len(env_list) == 1, f"Number of unwrapped base_env is not 1: {len(env_list)}!"
		env = env_list[0].env

		env_type = env.config['scene']
		
		episode.custom_metrics["episode_robot_energy"] = env.current_episode_robot_energy_cost
		if 'relocate' in env_type:
			episode.custom_metrics["episode_pushing_energy"] = env.current_episode_pushing_energy_translation + env.current_episode_pushing_energy_rotation


		if episode.last_info_for()['success']:
			episode.custom_metrics["success_rate"] = 1.
			episode.custom_metrics["succeed_episode_robot_energy"] = episode.custom_metrics["episode_robot_energy"]
			if 'relocate' in env_type:
				episode.custom_metrics["succeed_episode_pushing_energy"] = episode.custom_metrics["episode_pushing_energy"]
		else:
			episode.custom_metrics["success_rate"] = 0.
			episode.custom_metrics["succeed_episode_robot_energy"] = 0
			if 'relocate' in env_type:
				episode.custom_metrics["succeed_episode_pushing_energy"] = 0
				
class CustomLogger(LoggerCallback):
    def on_trial_start(self, iteration, trials, trial, **info):
        source_env_config_file = os.path.join(config_path, trial.config['env_config']['config_file'])
        logged_env_config_file = os.path.join(trial.logdir, trial.config['env_config']['config_file'])
        copyfile(source_env_config_file, logged_env_config_file)