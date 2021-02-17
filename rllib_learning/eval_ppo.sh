#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_4d6ad_00000_0_2021-02-16_17-41-49/checkpoint_175/checkpoint-175 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"turtlebot_relocate.yaml\", \
\"mode\": \"gui\", \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 50 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_4d6ad_00000_0_2021-02-16_17-41-49/rollouts.pkl \
--no-render


