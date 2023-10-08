from physics.gen.mass_matrix import mass_matrix
from physics.gen.bias_forces import bias_forces

from physics.simulate import step
from physics.visualize import animate

import jax
import jax.numpy as jnp
import jax.profiler
from clu import metrics
import flax
from flax import linen as nn
from flax import struct
import optax

import jax.experimental.host_callback

import shutil

from einops import einops, einsum
import matplotlib.pyplot as plt

from embeds import EmbeddingLayer
from training.rollout import collect_rollout
from training.vibe_state import (
    VibeState,
    TrainConfig,
)
from training.nets import (
    StateEncoder,
    ActionEncoder,
    TransitionModel,
    StateDecoder,
    ActionDecoder,
    encoded_state_dim,
    encoded_action_dim,
)

from training.train import train_step, dump_to_wandb
from policy import random_policy  # , max_dist_policy

import timeit

import os

from contextlib import redirect_stdout

import wandb

# from jax import config
# config.update("jax_disable_jit", True)

# Generate random key
key = jax.random.PRNGKey(1)

### Set up physics sim stuff
rng, key = jax.random.split(key)

mass_matrix = jax.jit(mass_matrix)
bias_forces = jax.jit(bias_forces)

mass_config = jnp.array([1.0, 0.25, 0.25, 0.04, 0.01, 0.01])
shape_config = jnp.array([1.0, 0.25, 0.25])

rng, key = jax.random.split(key)
home_q = jnp.array([0, 0, -0.0, 0.5, -0.5, -0.5, 0.5], dtype=jnp.float32)
start_q = home_q

qd = jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)


### Set up RL stuff

learning_rate = float(1e-5)
every_k = 1

vibe_config = TrainConfig(
    learning_rate=learning_rate,
    optimizer=optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(10.0),
            optax.lion(learning_rate=learning_rate),
        ),
        every_k_schedule=every_k,
    ),
    
    state_encoder=StateEncoder(),
    action_encoder=ActionEncoder(),
    transition_model=TransitionModel(1e4, 256),
    state_decoder=StateDecoder(),
    action_decoder=ActionDecoder(),
    
    batch_size=256,
    traj_per_rollout=2048,
    reconstruction_weight=1.0,
    forward_weight=1.0,
    rollout_length=5.0,
    dt=0.02,
    substep=2,
)

rng, key = jax.random.split(key)
vibe_state = VibeState.init(rng, vibe_config)

action_bounds = jnp.array([0.5, 0.5, 0.5, 0.5])


def policy(key, q, qd):
    state = jnp.concatenate([q, qd], axis=-1)
    target_state = jnp.concatenate([home_q, jnp.zeros_like(home_q)], axis=-1)

    rng, key = jax.random.split(key)
    # action = max_dist_policy(
    #     rng,
    #     state_encoder_state,
    #     action_decoder_state,
    #     transition_model_state,
    #     state,
    #     window=8,  # 128,
    #     dt=dt,
    # )
    action = random_policy(rng, action_bounds)

    clipped_action = jnp.clip(action, -5.0, 5.0)

    return clipped_action


policy = jax.tree_util.Partial(policy)


rng, key = jax.random.split(key)
rngs = jax.random.split(rng, start_q.shape[:-1])

# rollout_result = None

info_folder = "infos"
jax_log_path = os.path.join(info_folder, "jax_log.txt")

loss_weights = {
    "state_encoder": {
        "reconstruction": 1.0,
        "forward": 0.1,
        "smoothness": 0.0,
        "dispersion": 0.0,
    },
    "action_encoder": {
        "reconstruction": 1.0,
        "forward": 0.1,
        "smoothness": 0.0,
        "dispersion": 0.0,
        "condensation": 0.0,
    },
    "transition_model": {
        "forward": 1.0,
    },
    "state_decoder": {
        "reconstruction": 1.0,
    },
    "action_decoder": {
        "reconstruction": 1.0,
    },
}

rollouts = 1024
trajectories_per_rollout = 4096
epochs = 64
minibatch = 256

wandb.init(
    project="go_by_vibes",
    config=vibe_config.make_dict(),
    # mode="disabled",
)


def dump_infos_for_tap(tap_pack, _):
    infos, rollout = tap_pack
    dump_infos([info_folder], infos, rollout, every_k)


def dump_to_wandb_for_tap(tap_pack, _):
    infos, chunk_i = tap_pack
    dump_to_wandb(infos, chunk_i, every_k)


def do_rollout(carry_pack, _):
    (key, rollout_i, vibe_state) = carry_pack

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectories_per_rollout)

    rollout_result = jax.vmap(collect_rollout, in_axes=((None,) * 8 + (0,)))(
        start_q,
        qd,
        mass_config,
        shape_config,
        policy,
        vibe_config.dt,
        vibe_config.substep,
        int(vibe_config.rollout_length / vibe_config.dt),
        rngs,
    )

    # from jax import config
    # config.update("jax_disable_jit", True)

    def do_epoch(carry_pack, _):
        (
            key,
            epoch,
            vibe_state,
        ) = carry_pack

        # shuffle the data
        rng, key = jax.random.split(key)
        # Make a shuffled list of indices
        indices = jax.random.permutation(rng, rollout_result[0].shape[0])
        states = rollout_result[0][indices]
        actions = rollout_result[1][indices]

        jax.experimental.host_callback.id_tap(
            lambda epoch, _: print(f"Epoch {epoch}"), epoch
        )

        def process_batch(carry, rollout_result_batch):
            (
                key,
                chunk_i,
                vibe_state,
            ) = carry

            rollout_result_batch = (
                rollout_result_batch[0],
                rollout_result_batch[1][:, :-1],
            )

            rng, key = jax.random.split(key)
            (
                vibe_state,
                loss_infos,
            ) = jax.jit(train_step)(
                rng,
                vibe_state,
                vibe_config,
                rollout_result_batch,
                action_bounds,
            )

            msg = None

            jax.experimental.host_callback.id_tap(
                lambda chunk, _: print(f"Chunk {chunk}"), chunk_i
            )

            jax.experimental.host_callback.id_tap(
                dump_to_wandb_for_tap, (loss_infos, chunk_i)
            )

            return (key, chunk_i + 1, vibe_state), (msg, loss_infos)

        states_batched = einops.rearrange(
            states, "(r b) t d -> r b t d", b=(minibatch // every_k)
        )
        actions_batched = einops.rearrange(
            actions, "(r b) t d -> r b t d", b=(minibatch // every_k)
        )

        rollout_results_batched = (states_batched, actions_batched)

        rng, key = jax.random.split(key)
        init = (
            rng,
            0,
            vibe_state,
        )

        (
            _,
            _,
            vibe_state,
        ), (
            msg,
            loss_infos,
        ) = jax.lax.scan(process_batch, init, rollout_results_batched)

        return (key, epoch + 1, vibe_state), loss_infos

        # jax.profiler.save_device_memory_profile("memory.prof")

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, epochs)

    rng, key = jax.random.split(key)
    init = (rng, 0, vibe_state)

    jax.experimental.host_callback.id_tap(
        lambda rollout, _: print(f"Rollout {rollout}"), rollout_i
    )

    (_, _, vibe_state), infos = jax.lax.scan(do_epoch, init, None, length=epochs)

    # jax.experimental.host_callback.id_tap(dump_infos_for_tap, (infos, rollout_i))

    return (key, rollout_i + 1, vibe_state), _


rng, key = jax.random.split(key)
init = (rng, jnp.array(0), vibe_state)

(_, _, vibe_state), _ = jax.lax.scan(do_rollout, init, None, length=rollouts)


jax.debug.print("Done!")

# for i in range(16):
#     ani = animate(rollout_result[0][i, ..., :7], shape_config=shape_config, dt=dt)
#     plt.show()

# pass
