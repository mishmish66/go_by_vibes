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

from policy import random_policy, make_optimized_actions

from .rollout import collect_rollout


def evaluate_actor(
    key,
    start_state,
    env_cls,
    vibe_state: VibeState,
    vibe_config: TrainConfig,
    target_q=1.0,
    update_steps=2048,
):
    horizon = vibe_config.rollout_length

    rng, key = jax.random.split(key)
    latent_start_state = encode_state(key, start_state, vibe_state, vibe_config)

    def cost_func(state, action):
        state_cost = jnp.abs(state[0] - target_q)
        action_cost = 0.01 * jnp.linalg.norm(action, ord=1)

        return state_cost + action_cost

    def traj_cost_func(states, actions):
        return jnp.mean(jax.vmap(cost_func)(states, actions))

    def latent_traj_cost_func(key, latent_states, latent_actions):
        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, latent_states.shape[0])
        states = jax.vmap(decode_state, (0, 0, None, None))(
            rngs, latent_states, vibe_state, vibe_config
        )
        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, latent_actions.shape[0])
        actions = jax.vmap(decode_action, (0, 0, 0, None, None))(
            rngs, latent_actions, latent_states, vibe_state, vibe_config
        )

        return jnp.sum(traj_cost_func(states, actions))

    def latent_action_plan_cost_func(
        latent_actions,
        latent_start_state,
        vibe_state,
        vibe_config,
        key,
    ):
        rng, key = jax.random.split(key)
        latent_states_prime = infer_states(
            rng, latent_start_state, latent_actions, vibe_state, vibe_config
        )
        latent_states = jnp.concatenate(
            [latent_start_state[None], latent_states_prime], axis=0
        )[:-1]

        return latent_traj_cost_func(key, latent_states, latent_actions)

    rng, key = jax.random.split(key)
    loc_random_policy = jax.tree_util.Partial(random_policy)
    random_states, random_actions = collect_rollout(
        start_state,
        loc_random_policy,
        env_cls,
        vibe_state,
        vibe_config,
        rng,
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, random_states.shape[0])
    latent_random_states = jax.vmap(encode_state, (0, 0, None, None))(
        rngs, random_states, vibe_state, vibe_config
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, random_actions.shape[0])
    latent_random_actions = jax.vmap(encode_action, (0, 0, 0, None, None))(
        rngs, random_actions, latent_random_states, vibe_state, vibe_config
    )

    # stepsize = 0.125

    # def scanf(carry, _):
    #     latent_actions, key, i = carry

    #     cost, grad = jax.value_and_grad(latent_action_plan_cost_func, 1)(
    #         key,
    #         latent_actions,
    #     )

    #     latent_actions = latent_actions - stepsize * grad

    #     return (latent_actions, key, i + 1), cost

    # rng, key = jax.random.split(key)
    # init = (
    #     latent_random_actions,
    #     rng,
    #     0,
    # )

    # # Now scan
    # (result_latent_action_plan, _, _), costs = jax.lax.scan(
    #     scanf,
    #     init,
    #     None,
    #     update_steps,
    # )

    # actor = PresetActor(result_latent_action_plan)

    rng, key = jax.random.split(key)
    actor, costs = make_optimized_actions(
        rng,
        start_state,
        latent_action_plan_cost_func,
        vibe_state,
        vibe_config,
        env_cls,
    )

    rng, key = jax.random.split(key)
    result_states, result_actions = collect_rollout(
        start_state,
        actor,
        env_cls,
        vibe_state,
        vibe_config,
        rng,
    )

    final_cost = traj_cost_func(result_states, result_actions)

    info = Infos.init()
    info = info.add_plain_info("final_cost", final_cost)
    info = info.add_plain_info("starting expected cost", costs[0])
    info = info.add_plain_info("mid expected cost", costs[costs.shape[0] // 2])

    min_idx = jnp.argmin(costs)
    max_idx = jnp.argmax(costs)
    info = info.add_plain_info("min expected cost index", min_idx)
    info = info.add_plain_info("max expected cost index", max_idx)
    info = info.add_plain_info("min expected cost", costs[min_idx])
    info = info.add_plain_info("final expected cost", costs[-1])

    return (result_states, result_actions), info
