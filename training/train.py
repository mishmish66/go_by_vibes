import jax
from jax import numpy as jnp

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
        loss_per_traj, infos_per_traj = jax.vmap(composed_loss, (0, 0, 0, None, None))(
            rngs,
            rollout_result[0],
            rollout_result[1],
            updated_vibe_state,
            train_config,
        )
        
        infos = jax.tree_map(lambda x: jnp.mean(x, axis=0), infos_per_traj)
        loss = jnp.mean(loss_per_traj, axis=0)
        return loss, infos

    (
        vibe_grad,
        loss_infos,
    ) = jax.grad(
        loss_for_grad, has_aux=True
    )(vibe_state.extract_params(), rng)
    
    jax.debug.print("reconstruction_loss {}", loss_infos["reconstruction_loss"])

    vibe_state.apply_gradients(vibe_grad, train_config)

    return (
        vibe_state,
        loss_infos,
    )


def dump_to_wandb(infos, chunk_i, every_k):
    if chunk_i % every_k == 0:
        wandb.log(infos)
