#!/bin/sh

./rollout.py ~/ray_results/PPO_one_box_state_no_energy/PPO_OpenRoomEnvironmentRLLIB_a0b12_00000_0_2021-03-15_12-14-13/checkpoint_175/checkpoint-175 \
--config "{\"env_config\": {\"env\": \"relocate\", \
\"config_file\": \"fetch_relocate.yaml\", \
\"mode\": \"headless\", \
\"frame_stack\": 0, \
\"device_idx\": 0 },\
\"explore\": \"False\",\
\"num_workers\": 0}" \
--run PPO --env openvroom-v0 --episodes 30 --out ~/ray_results/PPO_one_box_state_no_energy/PPO_OpenRoomEnvironmentRLLIB_a0b12_00000_0_2021-03-15_12-14-13/rollouts.pkl \
--no-render


