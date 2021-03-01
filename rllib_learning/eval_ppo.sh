#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_b84f4_00000_0_2021-02-28_17-01-56/checkpoint_185/checkpoint-185 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 4, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 30 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_b84f4_00000_0_2021-02-28_17-01-56/rollouts.pkl \
--no-render


