#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_d083a_00000_0_2021-05-23_03-01-16/checkpoint_000250/checkpoint-250 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate_two_band_region_reverse.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 15 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_d083a_00000_0_2021-05-23_03-01-16/rollouts.pkl \
--no-render


