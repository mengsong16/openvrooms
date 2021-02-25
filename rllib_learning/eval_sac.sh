#!/bin/sh

rllib rollout ~/ray_results/SAC/SAC_OpenRoomEnvironmentRLLIB_53498_00000_0_2021-02-25_13-35-05/checkpoint_1049/checkpoint-1049 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run SAC --env openvroom-v0 --episodes 50 --out ~/ray_results/SAC/SAC_OpenRoomEnvironmentRLLIB_53498_00000_0_2021-02-25_13-35-05/rollouts.pkl \
--no-render


