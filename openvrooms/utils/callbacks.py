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

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv,
                       policies: Dict[str, Policy], episode: MultiAgentEpisode,
                       env_index: int, **kwargs):
        env_list = base_env.get_unwrapped()
        assert len(env_list) == 1, f"Number of unwrapped base_env is not 1: {len(env_list)}!"
        env = env_list[0].env
        
        episode.custom_metrics["episode_robot_energy"] = env.current_episode_robot_energy_cost
        episode.custom_metrics["episode_pushing_energy"] = env.current_episode_pushing_energy_translation
