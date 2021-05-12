#!/bin/sh

rllib rollout ~/ray_results/PPO_v4/PPO_OpenRoomEnvironmentRLLIB_e5ce7_00000_0_2021-04-18_18-02-22/checkpoint_000200/checkpoint-200 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate_short.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 30 --out ~/ray_results/PPO_v4/PPO_OpenRoomEnvironmentRLLIB_e5ce7_00000_0_2021-04-18_18-02-22/rollouts.pkl \
--no-render


