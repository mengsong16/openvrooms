#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_6221f_00000_0_2021-03-01_14-42-20/checkpoint_175/checkpoint-175 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 30 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_6221f_00000_0_2021-03-01_14-42-20/rollouts.pkl \
--no-render


