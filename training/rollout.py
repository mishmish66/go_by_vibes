import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node

from einops import rearrange


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
        action = policy(rng, state, vibe_state, vibe_config)
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
