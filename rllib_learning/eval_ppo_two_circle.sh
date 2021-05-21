#!/bin/sh

rllib rollout ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_3fc27_00000_0_2021-05-02_19-42-49/checkpoint_000250/checkpoint-250 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate_circle_two_box.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 30 --out ~/ray_results/PPO/PPO_OpenRoomEnvironmentRLLIB_3fc27_00000_0_2021-05-02_19-42-49/rollouts.pkl \
--no-render

