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
    n_gaussian_samples=16,
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
        
        scaled_gated_losses, loss_infos  = losses.scale_gate_info(train_config)

        infos = Infos.merge(infos, loss_infos)

        # return total_loss, infos
        return scaled_gated_losses.to_list(), infos
    
    def loss_for_grad_good(vibe_params, key):
        losses, infos = loss_for_grad(vibe_params, key)
        
        losses = Losses.from_list(losses)
        
        losses = Losses.init(
            reconstruction_loss=losses.reconstruction_loss,
            forward_loss = losses.forward_loss,
            smoothness_loss=losses.smoothness_loss,
        )
        
        return losses.to_list(), infos

    vibe_grads_bad, _ = jax.jacrev(loss_for_grad, has_aux=True)(
        vibe_state.extract_params(), rng
    )

    vibe_grads, (loss_infos) = jax.jacrev(loss_for_grad_good, has_aux=True)(
        vibe_state.extract_params(), rng
    )

    def concat_leaves(tree):
        if isinstance(tree, dict):
            return jnp.concatenate([concat_leaves(child) for child in tree.values()])
        elif isinstance(tree, jax.Array):
            return jnp.ravel(tree)
        else:
            jnp.array([])
            
    grad_norms = [
        jnp.linalg.norm(jnp.nan_to_num(concat_leaves(grad))) for grad in vibe_grads
    ]
    
    bad_grad_norms = [
        jnp.linalg.norm(jnp.nan_to_num(concat_leaves(grad))) for grad in vibe_grads_bad
    ]
    
    grad_diff_norms = [
        jnp.linalg.norm(jnp.nan_to_num(concat_leaves(grad)) - jnp.nan_to_num(concat_leaves(bad_grad)))
        for grad, bad_grad in zip(vibe_grads, vibe_grads_bad)
    ]
        
    loss_infos = loss_infos.add_plain_info("reconstruction_grad_norm", grad_norms[0])
    loss_infos = loss_infos.add_plain_info("forward_grad_norm", grad_norms[1])
    loss_infos = loss_infos.add_plain_info("smoothness_grad_norm", grad_norms[2])
    loss_infos = loss_infos.add_plain_info("dispersion_grad_norm", grad_norms[3])
    loss_infos = loss_infos.add_plain_info("condensation_grad_norm", grad_norms[4])
        
    loss_infos = loss_infos.add_plain_info("bad_reconstruction_grad_norm", bad_grad_norms[0])
    loss_infos = loss_infos.add_plain_info("bad_forward_grad_norm", bad_grad_norms[1])
    loss_infos = loss_infos.add_plain_info("bad_smoothness_grad_norm", bad_grad_norms[2])
    loss_infos = loss_infos.add_plain_info("bad_dispersion_grad_norm", bad_grad_norms[3])
    loss_infos = loss_infos.add_plain_info("bad_condensation_grad_norm", bad_grad_norms[4])
        
    loss_infos = loss_infos.add_plain_info("reconstruction_grad_diff_norm", grad_diff_norms[0])
    loss_infos = loss_infos.add_plain_info("forward_grad_diff_norm", grad_diff_norms[1])
    loss_infos = loss_infos.add_plain_info("smoothness_grad_diff_norm", grad_diff_norms[2])
    loss_infos = loss_infos.add_plain_info("dispersion_grad_diff_norm", grad_diff_norms[3])
    loss_infos = loss_infos.add_plain_info("condensation_grad_diff_norm", grad_diff_norms[4])
    
    vibe_grad = jax.tree_map(lambda *x: jnp.sum(jnp.stack(x), axis=0), *vibe_grads)

    total_grad = concat_leaves(vibe_grad)
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
