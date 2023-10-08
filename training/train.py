import jax
from jax import numpy as jnp

from jax.experimental.host_callback import id_tap

from .vibe_state import VibeState, TrainConfig
from .loss import composed_loss

import wandb


def train_step(
    key,
    vibe_state: VibeState,
    train_config: TrainConfig,
    rollout_result,
    action_bounds,
):
    """Train for a single step."""

    rng, key = jax.random.split(key)

    def loss_for_grad(vibe_params, key):
        updated_vibe_state = vibe_state.assign_dict(vibe_params)
        n_traj = rollout_result[0].shape[0]

        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, n_traj)
        losses_per_traj = jax.vmap(composed_loss, (0, 0, 0, None, None))(
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

        gate_value = jax.lax.stop_gradient(shaped_sigmoid_reconstruction_loss)

        # TODO: replace this with a better way of getting infos
        return (
            losses.reconstruction_loss + losses.forward_loss * gate_value,
            {
                **losses.make_dict(),
                "gate_value": gate_value,
            },
        )

    (
        vibe_grad,
        loss_infos,
    ) = jax.grad(
        loss_for_grad, has_aux=True
    )(vibe_state.extract_params(), rng)

    grad_norms = jax.tree_map(lambda x: jnp.linalg.norm(jnp.ravel(x)), vibe_grad)

    id_tap(
        lambda loss_infos, _: print(
            f"reconstruction_loss: {loss_infos['reconstruction_loss']}\n"
            + f"forward_loss: {loss_infos['forward_loss']}\n"
            + f"gate_value: {loss_infos['gate_value']}\n"
        ),
        loss_infos,
    )

    # print grads
    # id_tap(
    #     lambda grad_norms, _: print(
    #         f"state_encoder_params: {grad_norms['state_encoder_params']}\n"
    #         + f"action_encoder_params: {grad_norms['action_encoder_params']}\n"
    #         + f"transition_model_params: {grad_norms['transition_model_params']}\n"
    #         + f"state_decoder_params: {grad_norms['state_decoder_params']}\n"
    #         + f"action_decoder_params: {grad_norms['action_decoder_params']}\n"
    #     ),
    #     grad_norms,
    # )
    
    # jax.debug.breakpoint()

    vibe_state = vibe_state.apply_gradients(vibe_grad, train_config)

    return (
        vibe_state,
        loss_infos,
    )


def dump_to_wandb(infos, chunk_i, every_k):
    if chunk_i % every_k == 0:
        wandb.log(infos)
