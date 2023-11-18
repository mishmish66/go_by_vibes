import jax
from jax import numpy as jnp

from jax.experimental.host_callback import id_tap

from .vibe_state import VibeState, TrainConfig
from .loss import (
    composed_random_index_losses,
    unordered_losses,
    Losses,
    make_gate_value,
)

from .infos import Infos

from einops import rearrange

import wandb


def train_step(
    key,
    vibe_state: VibeState,
    train_config: TrainConfig,
    rollout_result,
    n_random_index_samples=1,
    n_gaussian_samples=1,
):
    """Train for a single step."""

    rng, key = jax.random.split(key)

    def loss_for_grad(vibe_params, key):
        updated_vibe_state = vibe_state.assign_dict(vibe_params)
        n_traj = rollout_result[0].shape[0]

        rng, key = jax.random.split(key)
        rngs_per_traj = jax.random.split(rng, n_traj)
        rngs = jax.vmap(jax.random.split, (0, None))(
            rngs_per_traj, n_random_index_samples
        )
        losses_per_traj_per_random_index, infos_per_traj_per_random_index = jax.vmap(
            jax.vmap(composed_random_index_losses, (0, None, None, None, None)),
            (0, 0, 0, None, None),
        )(
            rngs,
            rollout_result[0],
            rollout_result[1],
            updated_vibe_state,
            train_config,
        )

        prev_states = rollout_result[0][:, :-1, :]

        flat_states = rearrange(prev_states, "b t d -> (b t) d")
        flat_actions = rearrange(rollout_result[1], "b t d -> (b t) d")

        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, 1)
        (
            losses_per_traj_per_gauss_sample,
            infos_per_traj_per_gauss_sample,
        ) = jax.vmap(
            unordered_losses,
            in_axes=(0, None, None, None, None),
        )(
            rngs,
            flat_states,
            flat_actions,
            updated_vibe_state,
            train_config,
        )

        def process_losses(losses):
            return jax.tree_map(
                lambda x: jnp.mean(x, axis=0),
                losses,
            )

        random_index_losses = process_losses(
            process_losses(losses_per_traj_per_random_index)
        )
        whole_traj_losses = process_losses(losses_per_traj_per_gauss_sample)

        losses = Losses.merge(random_index_losses, whole_traj_losses)

        infos_per_traj_per_comp = Infos.merge(
            infos_per_traj_per_random_index,
            infos_per_traj_per_gauss_sample,
        )

        infos = infos_per_traj_per_comp.condense()

        scaled_gated_losses, loss_infos = losses.scale_gate_info(train_config)

        infos = Infos.merge(infos, loss_infos)

        # return total_loss, infos
        return scaled_gated_losses, infos

    vibe_grads, (loss_infos) = jax.jacrev(loss_for_grad, has_aux=True)(
        vibe_state.extract_params(), rng
    )

    forward_loss = loss_infos.loss_infos["forward_loss"]

    forward_blend_gate = make_gate_value(
        forward_loss,
        train_config.forward_blend_gate_sharpness,
        train_config.forward_blend_gate_center,
    )

    def scale_grad(grad, gate):
        return jax.tree_map(lambda x: x * gate, grad)

    forward_loss_grads = vibe_grads.forward_loss
    forward_loss_grads["action_encoder_params"] = scale_grad(
        forward_loss_grads["action_encoder_params"], forward_blend_gate
    )
    forward_loss_grads["action_decoder_params"] = scale_grad(
        forward_loss_grads["action_decoder_params"], forward_blend_gate
    )
    forward_loss_grads["state_encoder_params"] = scale_grad(
        forward_loss_grads["state_encoder_params"], forward_blend_gate
    )
    forward_loss_grads["state_decoder_params"] = scale_grad(
        forward_loss_grads["state_decoder_params"], forward_blend_gate
    )

    vibe_grads = vibe_grads.replace(forward_loss=forward_loss_grads)

    def concat_leaves(tree):
        if isinstance(tree, dict):
            return jnp.concatenate([concat_leaves(child) for child in tree.values()])
        elif isinstance(tree, jax.Array):
            return jnp.ravel(tree)
        else:
            jnp.array([])

    grad_norms = Losses.from_list(
        [
            jnp.linalg.norm(jnp.nan_to_num(concat_leaves(grad)))
            for grad in vibe_grads.to_list()
        ]
    )

    loss_infos = loss_infos.add_plain_info(
        "reconstruction_grad_norm", grad_norms.reconstruction_loss
    )
    loss_infos = loss_infos.add_plain_info("forward_grad_norm", grad_norms.forward_loss)
    loss_infos = loss_infos.add_plain_info(
        "smoothness_grad_norm", grad_norms.smoothness_loss
    )
    loss_infos = loss_infos.add_plain_info(
        "dispersion_grad_norm", grad_norms.dispersion_loss
    )
    loss_infos = loss_infos.add_plain_info(
        "condensation_grad_norm", grad_norms.condensation_loss
    )

    vibe_grad = jax.tree_map(
        lambda *x: jnp.sum(jnp.stack(x), axis=0), *(vibe_grads.to_list())
    )

    total_grad = concat_leaves(vibe_grad)

    grad_nan_portion = jnp.mean(jnp.isnan(total_grad))
    loss_infos = loss_infos.add_plain_info("grad nan portion", grad_nan_portion)

    total_grad = jnp.nan_to_num(total_grad)
    total_grad_norm = jnp.linalg.norm(total_grad)

    vibe_state = vibe_state.apply_gradients(vibe_grad, train_config)

    loss_infos = loss_infos.add_plain_info("total_grad_norm", total_grad_norm)

    return vibe_state, loss_infos


def dump_to_wandb(infos, rollout_i, epoch_i, chunk_i, train_config: TrainConfig):
    steps_per_epoch = train_config.traj_per_rollout // train_config.batch_size
    step = (
        rollout_i * train_config.epochs
        + epoch_i * steps_per_epoch
        + chunk_i // train_config.every_k
    )
    if chunk_i % train_config.every_k == 0:
        wandb.log(infos)
