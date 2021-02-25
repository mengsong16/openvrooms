#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_96e72_00000_0_2021-02-24_11-22-09/checkpoint_60/checkpoint-60 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 50 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_96e72_00000_0_2021-02-24_11-22-09/rollouts.pkl \
--no-render


