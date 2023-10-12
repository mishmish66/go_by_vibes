import jax
from jax import numpy as jnp

from jax.experimental.host_callback import id_tap

from .vibe_state import VibeState, TrainConfig
from .loss import composed_loss

from .infos import Infos

from einops import rearrange

import wandb


def train_step(
    key,
    vibe_state: VibeState,
    train_config: TrainConfig,
    rollout_result,
):
    """Train for a single step."""

    rng, key = jax.random.split(key)

    def loss_for_grad(vibe_params, key):
        updated_vibe_state = vibe_state.assign_dict(vibe_params)
        n_traj = rollout_result[0].shape[0]

        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, n_traj)
        losses_per_traj, infos_per_traj = jax.vmap(
            composed_loss, (0, 0, 0, None, None)
        )(
            rngs,
            rollout_result[0],
            rollout_result[1],
            updated_vibe_state,
            train_config,
        )

        losses = jax.tree_map(lambda x: jnp.mean(x, axis=0), losses_per_traj)

        shaped_sigmoid_reconstruction_loss = 1 / (
            1 + jnp.exp(losses.reconstruction_loss + 50)
        )

        infos = Infos.init(
            loss_infos=jax.tree_map(
                lambda x: jnp.mean(x, axis=0), infos_per_traj.loss_infos
            ),
            plain_infos=jax.tree_map(
                lambda x: rearrange(x, "t n ... -> (t n) ..."),
                infos_per_traj.plain_infos,
            ),
            masked_infos=jax.tree_map(
                lambda x: rearrange(x, "t n ... -> (t n) ..."),
                infos_per_traj.masked_infos,
            ),
        )

        gate_value = jax.lax.stop_gradient(shaped_sigmoid_reconstruction_loss)

        infos = infos.add_plain_info("gate_value", gate_value)

        return losses.reconstruction_loss + losses.forward_loss * gate_value, infos

    (
        vibe_grad,
        loss_infos,
    ) = jax.grad(
        loss_for_grad, has_aux=True
    )(vibe_state.extract_params(), rng)

    def concat_leaves(tree):
        if isinstance(tree, dict):
            return jnp.concatenate([concat_leaves(child) for child in tree.values()])
        elif isinstance(tree, jax.Array):
            return jnp.ravel(tree)
        else:
            jnp.array([])

    total_grad = concat_leaves(vibe_grad)
    total_grad_norm = jnp.linalg.norm(total_grad)

    # id_tap(
    #     lambda loss_infos, _: print(
    #         f"reconstruction_loss: {loss_infos['reconstruction_loss']}\n"
    #         + f"forward_loss: {loss_infos['forward_loss']}\n"
    #         + f"gate_value: {loss_infos['gate_value']}\n"
    #     ),
    #     loss_infos,
    # )

    vibe_state = vibe_state.apply_gradients(vibe_grad, train_config)

    loss_infos = loss_infos.add_plain_info("total_grad_norm", total_grad_norm)

    return vibe_state, loss_infos
