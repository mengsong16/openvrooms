
from gym.envs.registration import register

from openvrooms.envs.relocate_env import RelocateEnv
from openvrooms.envs.navigate_env import NavigateEnv

# register envs to gym 
register(
    id='openrelocate-v0',
    entry_point='openvrooms.envs.relocate_env:RelocateEnv',
)

register(
    id='opennavigate-v0',
    entry_point='openvrooms.envs.navigate_env:NavigateEnv',
)

