#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_c8337_00000_0_2021-02-08_17-33-16/checkpoint_62/checkpoint-62 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"turtlebot_relocate.yaml\", \
\"mode\": \"gui\", \
\"device_idx\": 0 },\
\"explore\": \"False\"}" \
--run PPO --env openvroom-v0 --episodes 50 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_c8337_00000_0_2021-02-08_17-33-16/rollouts.pkl \
--no-render


