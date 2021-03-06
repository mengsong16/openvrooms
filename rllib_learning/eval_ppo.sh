#!/bin/sh

rllib rollout ~/ray_results/PPO_original_short/PPO_OpenRoomEnvironmentRLLIB_7f9d3_00000_0_2021-05-19_02-49-26/checkpoint_000250/checkpoint-250 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate_short.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 50 --out ~/ray_results/PPO_original_short/PPO_OpenRoomEnvironmentRLLIB_7f9d3_00000_0_2021-05-19_02-49-26/rollouts.pkl \
--no-render \
--save-info


