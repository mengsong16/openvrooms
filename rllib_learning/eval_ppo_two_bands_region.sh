#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_ea22a_00000_0_2021-05-23_00-45-59/checkpoint_000250/checkpoint-250 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate_two_band_region.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 15 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_ea22a_00000_0_2021-05-23_00-45-59/rollouts.pkl \
--no-render


