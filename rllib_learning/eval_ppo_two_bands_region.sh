#!/bin/sh

rllib rollout ~/ray_results/two-band-paper/PPO_OpenRoomEnvironmentRLLIB_f155f_00000_0_2021-05-27_17-59-23/checkpoint_000300/checkpoint-300 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate_two_band_region.yaml\", \
\"mode\": \"gui\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 30 --out ~/ray_results/two-band-paper/PPO_OpenRoomEnvironmentRLLIB_f155f_00000_0_2021-05-27_17-59-23/rollouts.pkl \
--no-render


