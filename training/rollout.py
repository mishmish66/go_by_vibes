import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node

from einops import rearrange

from .vibe_state import VibeState, TrainConfig

from .inference import (
    encode_state,
    encode_action,
    decode_state,
    decode_action,
    infer_states,
)

from dataclasses import dataclass

from .infos import Infos


def collect_rollout(
    start_state,
    policy,
    env_cls,
    vibe_state,
    vibe_config,
    key,
):
    # Collect a rollout of physics data
    def scanf(carry, _):
        state, key, i = carry

        rng, key = jax.random.split(key)
        action = policy(rng, state, i, vibe_state, vibe_config)
        next_state = env_cls.step(state, action, vibe_config.env_config)

        return (next_state, key, i + 1), (state, action)

    rng, key = jax.random.split(key)
    _, (states, actions) = jax.lax.scan(
        scanf,
        (start_state, rng, 0),
        None,
        vibe_config.rollout_length,
    )

    return states, actions


@jax.tree_util.register_pytree_node_class
@dataclass
class PresetActor:
    actions: all

    def tree_flatten(self):
        return (self.actions,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __call__(self, key, state, i, vibe_state, vibe_config):
        # encode the state
        rng, key = jax.random.split(key)
        latent_state = encode_state(rng, state, vibe_state, vibe_config)

        # decode the action
        latent_action = self.actions[i]
        rng, key = jax.random.split(key)
        action = decode_action(
            rng, latent_action, latent_state, vibe_state, vibe_config
        )
        return action