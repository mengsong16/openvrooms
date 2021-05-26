#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_9b7a2_00000_0_2021-05-26_12-21-44/checkpoint_000100/checkpoint-100 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate_outside_circle_two_box.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 40 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_9b7a2_00000_0_2021-05-26_12-21-44/rollouts.pkl \
--no-render \
--save-info


