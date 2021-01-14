
from openvrooms.envs.relocate_env import RelocateEnv
from gym.envs.registration import register

# register envs to gym 
register(
    id='openvrooms-v0',
    entry_point='openvrooms.envs.relocate_env:RelocateEnv',
)
