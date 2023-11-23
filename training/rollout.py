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
    init_policy_carry,
    env_cls,
    vibe_state,
    vibe_config,
    key,
):
    policy = jax.tree_util.Partial(policy)

    # Collect a rollout of physics data
    def scanf(carry, _):
        state, key, i, policy_carry = carry

        rng, key = jax.random.split(key)
        action, policy_carry = policy(
            rng, state, i, policy_carry, vibe_state, vibe_config
        )
        action = jnp.clip(
            action,
            a_min=vibe_config.env_config.action_bounds[..., 0],
            a_max=vibe_config.env_config.action_bounds[..., -1],
        )
        next_state = env_cls.step(state, action, vibe_config.env_config)

        return (next_state, key, i + 1, policy_carry), (state, action)

    rng, key = jax.random.split(key)
    _, (states, actions) = jax.lax.scan(
        scanf,
        (start_state, rng, 0, init_policy_carry),
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
