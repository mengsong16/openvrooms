#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_193d8_00000_0_2021-05-20_15-45-39/checkpoint_000300/checkpoint-300 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate_two_band_short.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 15 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_193d8_00000_0_2021-05-20_15-45-39/rollouts.pkl \
--no-render


