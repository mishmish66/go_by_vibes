from typing import Any
import flax
from flax import linen as nn


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
        x = nn.Dense(12, name="FC5")(x)
        return x


class Transition(nn.Module):
    @nn.compact
    def __call__(self, x) -> Any:
        x = nn.Dense(12, name="FC1")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC2")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC3")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC4")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC5")(x)
        x = nn.relu(x)
        x = nn.Dense(6, name="FC6")(x)
        return x
