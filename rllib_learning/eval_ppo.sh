#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_828c8_00000_0_2021-02-17_14-28-51/checkpoint_143/checkpoint-143 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"turtlebot_relocate.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 4, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 50 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_828c8_00000_0_2021-02-17_14-28-51/rollouts.pkl \
--no-render


