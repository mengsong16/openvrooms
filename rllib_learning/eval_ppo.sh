#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_5e675_00000_0_2021-02-25_13-42-33/checkpoint_457/checkpoint-457 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 50 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_5e675_00000_0_2021-02-25_13-42-33/rollouts.pkl \
--no-render


