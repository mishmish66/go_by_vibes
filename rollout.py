import jax
from jax import numpy as jnp
from jax.tree_util import register_pytree_node

from einops import rearrange

from physics import step


def collect_rollout(
    q_0,
    qd_0,
    mass_config,
    shape_config,
    policy,
    dt,
    substep: int,
    steps: int,
    key,
):
    # Collect a rollout of physics data
    def scanf(carry, _):
        q, qd, action, key, i = carry

        rng, key = jax.random.split(key)
        new_action = policy(
            q,
            qd,
            rng,
        )

        action = jax.lax.cond(
            jax.numpy.mod(i, substep) == 0,
            lambda: new_action,
            lambda: action,
        )

        control = jnp.concatenate([jnp.zeros(3), action])
        q, qd = step(q, qd, mass_config, shape_config, control, dt / substep)

        # jax.debug.print("Whomp!")
        return (q, qd, action, key, i + 1), (q, qd, action)

    jax.debug.print("Scanning!")

    _, (q_result_sub, qd_result_sub, action_result_sub) = jax.lax.scan(
        scanf,
        (q_0, qd_0, policy(q_0, qd_0, key), key, 0),
        None,
        steps * substep,
    )

    # Now transform that data through our models
    qs, qds, actions = (
        q_result_sub[::substep],
        qd_result_sub[::substep],
        action_result_sub[::substep],
    )

    states = rearrange([qs, qds], "i t d -> t (i d)")

    return states, actions


# def flatten_rollout_result(rollout_result):
#     return [
#         rollout_result.states,
#         rollout_result.actions,
#     ], None


# def unflatten_rollout_result(aux, flat_rollout_result):
#     return RolloutResult(*flat_rollout_result)


# register_pytree_node(
#     RolloutResult,
#     flatten_rollout_result,
#     unflatten_rollout_result,
# )
