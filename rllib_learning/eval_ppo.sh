#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_fe541_00000_0_2021-02-16_12-38-57/checkpoint_125/checkpoint-125 \
--config "{\"env_config\": {\"env\": \"navigate\", \
\"config_file\": \"turtlebot_navigate.yaml\", \
\"mode\": \"gui\", \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 50 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_fe541_00000_0_2021-02-16_12-38-57/rollouts.pkl \
--no-render


