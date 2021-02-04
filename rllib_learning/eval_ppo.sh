#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_d90c7_00000_0_2021-02-04_00-42-00/checkpoint_25/checkpoint-25 \
--config "{\"env_config\": {\"env\": \"navigate\", \
\"config_file\": \"turtlebot_navigate.yaml\", \
\"mode\": \"headless\", \
\"device_idx\": 0 },\
\"explore\": \"False\"}" \
--run PPO --env openvroom-v0 --episodes 50 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_d90c7_00000_0_2021-02-04_00-42-00/rollouts.pkl \
--no-render


