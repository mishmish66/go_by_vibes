import jax
from jax import numpy as jnp

from jax.experimental.host_callback import id_tap

from .vibe_state import VibeState, TrainConfig
from .loss import composed_random_index_losses, composed_whole_traj_losses, Losses

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
        rngs = jax.vmap(jax.random.split, (0, None))(rngs_per_traj, n_random_index_samples)
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

        rng, key = jax.random.split(key)
        rngs_per_traj = jax.random.split(rng, n_traj)
        rngs = jax.vmap(jax.random.split, (0, None))(rngs_per_traj, n_gaussian_samples)
        losses_per_traj_per_gauss_sample, infos_per_traj_per_gauss_sample = jax.vmap(
            jax.vmap(composed_whole_traj_losses, (0, None, None, None, None)),
            (0, 0, 0, None, None),
        )(
            rngs,
            rollout_result[0],
            rollout_result[1],
            updated_vibe_state,
            train_config,
        )

        def process_losses(losses):
            return jax.tree_map(
                lambda x: jnp.mean(jnp.mean(x, axis=0), axis=0),
                losses,
            )

        random_index_losses = process_losses(losses_per_traj_per_random_index)
        whole_traj_losses = process_losses(losses_per_traj_per_gauss_sample)

        losses = Losses.merge(random_index_losses, whole_traj_losses)

        shaped_sigmoid_reconstruction_loss = 1 / (
            1 + jnp.exp(losses.reconstruction_loss + 50)
        )

        def process_infos(infos):
            return Infos.init(
                loss_infos=jax.tree_map(
                    lambda x: jnp.mean(jnp.mean(x, axis=0), axis=0),
                    infos.loss_infos,
                ),
                plain_infos=jax.tree_map(
                    lambda x: rearrange(x, "t n ... -> (t n) ..."),
                    infos.plain_infos,
                ),
                masked_infos=jax.tree_map(
                    lambda x: rearrange(x, "t n ... -> (t n) ..."),
                    infos.masked_infos,
                ),
            )

        random_i_infos = process_infos(infos_per_traj_per_random_index)
        whole_traj_infos = process_infos(infos_per_traj_per_gauss_sample)

        infos = Infos.merge(random_i_infos, whole_traj_infos)

        gate_value = jax.lax.stop_gradient(shaped_sigmoid_reconstruction_loss)

        infos = infos.add_plain_info("gate_value", gate_value)

        return (
            losses.reconstruction_loss * train_config.reconstruction_weight
            + (
                losses.forward_loss
                * train_config.forward_weight
                + losses.smoothness_loss * train_config.smoothness_weight
                # + losses.dispersion_loss * train_config.dispersion_weight
                # + losses.condensation_loss * train_config.condensation_weight
            )
            * gate_value,
            infos,
        )

    (vibe_grad, loss_infos) = jax.grad(loss_for_grad, has_aux=True)(
        vibe_state.extract_params(), rng
    )

    def concat_leaves(tree):
        if isinstance(tree, dict):
            return jnp.concatenate([concat_leaves(child) for child in tree.values()])
        elif isinstance(tree, jax.Array):
            return jnp.ravel(tree)
        else:
            jnp.array([])

    total_grad = concat_leaves(vibe_grad)
    total_grad = jnp.nan_to_num(total_grad)
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


def dump_to_wandb(infos, rollout_i, epoch_i, chunk_i, train_config: TrainConfig):
    steps_per_epoch = train_config.traj_per_rollout // train_config.batch_size
    step = (
        rollout_i * train_config.epochs
        + epoch_i * steps_per_epoch
        + chunk_i // train_config.every_k
    )
    if chunk_i % train_config.every_k == 0:
        wandb.log(infos)
