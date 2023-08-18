from typing import Any

import flax
from flax import linen as nn

from einops import rearrange

encoded_state_dim = 32
encoded_action_dim = 16


class StateEncoder(nn.Module):
    @nn.compact
    def __call__(self, x) -> Any:
        x = nn.Dense(128, name="FC1")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC2")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC3")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC4")(x)
        x = nn.relu(x)
        x = nn.Dense(encoded_state_dim, name="FC5")(x)
        return x
    
class StateDecoder(nn.Module):
    @nn.compact
    def __call__(self, x) -> Any:
        x = nn.Dense(128, name="FC1")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC2")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC3")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC4")(x)
        x = nn.relu(x)
        x = nn.Dense(14, name="FC5")(x)
        return x

class ActionEncoder(nn.Module):
    @nn.compact
    def __call__(self, x) -> Any:
        x = nn.Dense(128, name="FC1")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC2")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC3")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC4")(x)
        x = nn.relu(x)
        x = nn.Dense(encoded_action_dim, name="FC5")(x)
        return x
    
class ActionDecoder(nn.Module):
    @nn.compact
    def __call__(self, x) -> Any:
        x = nn.Dense(128, name="FC1")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC2")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC3")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC4")(x)
        x = nn.relu(x)
        x = nn.Dense(4, name="FC5")(x)
        return x

class TransitionModel(nn.Module):
    @nn.compact
    def __call__(self, encoded_state, encoded_action) -> Any:
        x = rearrange([encoded_state, encoded_action], "s a -> (s a)")
        x = nn.Dense(256, name="FC1")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC2")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC3")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC4")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC5")(x)
        x = nn.relu(x)
        x = nn.Dense(2*encoded_state_dim, name="FC6")(x)
        return x
