#!/bin/sh

rllib rollout ~/ray_results/DQN/DQN_OpenRoomEnvironmentRLLIB_34801_00000_0_2021-02-03_15-47-41/checkpoint_100/checkpoint-100 \
--config "{\"env_config\": {\"env\": \"navigate\", \
\"config_file\": \"turtlebot_navigate.yaml\", \
\"mode\": \"gui\", \
\"device_idx\": 0 },\
\"explore\": \"False\"}" \
--run DQN --env openvroom-v0 --episodes 10 --out ~/ray_results/DQN/DQN_OpenRoomEnvironmentRLLIB_34801_00000_0_2021-02-03_15-47-41/rollouts.pkl \
--no-render


