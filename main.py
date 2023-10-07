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
from rollout import collect_rollout
from training import (
    create_train_state,
    train_step,
    compute_metrics,
    dump_infos,
    dump_to_wandb,
    make_info_msgs,
    merge_info_msgs,
)
from nets import (
    StateEncoder,
    ActionEncoder,
    TransitionModel,
    StateDecoder,
    ActionDecoder,
    encoded_state_dim,
    encoded_action_dim,
)
from policy import max_dist_policy, random_policy

import timeit

import os

from contextlib import redirect_stdout

import wandb

shutil.rmtree("infos", ignore_errors=True)

# Generate random key
key = jax.random.PRNGKey(1)

### Set up physics sim stuff
rng, key = jax.random.split(key)

mass_matrix = jax.jit(mass_matrix)
bias_forces = jax.jit(bias_forces)

mass_config = jnp.array([1.0, 0.25, 0.25, 0.04, 0.01, 0.01])
shape_config = jnp.array([1.0, 0.25, 0.25])

envs = 1024

rng, key = jax.random.split(key)
home_q = jnp.array([0, 0, -0.0, 0.5, -0.5, -0.5, 0.5], dtype=jnp.float32)
# home_q = jnp.array([0, 0, -0.0, -jnp.pi, -0.0, -jnp.pi, 0.0], dtype=jnp.float32)
start_q = home_q  # + jax.random.normal(rng, (envs, 7)) * 0.25

# rng, key = jax.random.split(key)
qd = jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
# qd = (
#     jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
#     + jax.random.normal(rng, (envs, 7)) * 0.1
# )

dt = 0.02
substep = 2
total_time = 2.5


### Set up RL stuff

state_embedding_layer = EmbeddingLayer(256)
action_embedding_layer = EmbeddingLayer(128)

# rng, key = jax.random.split(key)

# params = state_encoder.init(rng, q)
# z_state = state_encoder.apply(params, q)

state_embeddings = EmbeddingLayer(encoded_state_dim)
action_embeddings = EmbeddingLayer(encoded_action_dim)

state_encoder = StateEncoder()
action_encoder = ActionEncoder()
transition_model = TransitionModel(n=1e4, latent_dim=128)
state_decoder = StateDecoder()
action_decoder = ActionDecoder()

learning_rates = {
    "state_encoder": 1e-5,
    "action_encoder": 1e-5,
    "transition_model": 1e-6,
    "state_decoder": 1e-5,
    "action_decoder": 1e-5,
}

every_k = 8

rng, key = jax.random.split(key)
state_encoder_state = create_train_state(
    state_encoder,
    [14],
    rng,
    learning_rate=learning_rates["state_encoder"],
    every_k=every_k,
)

rng, key = jax.random.split(key)
action_encoder_state = create_train_state(
    action_encoder,
    [4, encoded_state_dim],
    rng,
    learning_rate=learning_rates["action_encoder"],
    every_k=every_k,
)

rng, key = jax.random.split(key)
transition_model_state = create_train_state(
    transition_model,
    [(16, encoded_state_dim), (16, encoded_action_dim), (16,)],
    rng,
    learning_rate=learning_rates["transition_model"],
    every_k=every_k,
)

rng, key = jax.random.split(key)
state_decoder_state = create_train_state(
    state_decoder,
    [encoded_state_dim],
    rng,
    learning_rate=learning_rates["state_decoder"],
    every_k=every_k,
)

rng, key = jax.random.split(key)
action_decoder_state = create_train_state(
    action_decoder,
    [encoded_action_dim, encoded_state_dim],
    rng,
    learning_rate=learning_rates["action_decoder"],
    every_k=every_k,
)

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

rollout_result = None

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

config = {
    "learning_rates": learning_rates,
    "loss_weights": loss_weights,
    "rollouts": rollouts,
    "trajectories_per_rollout": trajectories_per_rollout,
    "epochs": epochs,
    "minibatch": minibatch,
}

wandb.init(
    project="go_by_vibes",
    config=config,
    # mode="disabled",
)


def dump_infos_for_tap(tap_pack, _):
    infos, rollout = tap_pack
    dump_infos([info_folder], infos, rollout, every_k)
    
def dump_to_wandb_for_tap(tap_pack, _):
    infos, chunk_i = tap_pack
    dump_to_wandb(infos, chunk_i, every_k)


def do_rollout(carry_pack, _):
    (
        key,
        rollout_i,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
    ) = carry_pack

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectories_per_rollout)

    rollout_result = jax.vmap(collect_rollout, in_axes=((None,) * 8 + (0,)))(
        start_q,
        qd,
        mass_config,
        shape_config,
        policy,
        dt,
        substep,
        int(total_time / dt),
        rngs,
    )

    # from jax import config
    # config.update("jax_disable_jit", True)

    def do_epoch(carry_pack, _):
        (
            key,
            epoch,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
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
                state_encoder_state,
                action_encoder_state,
                transition_model_state,
                state_decoder_state,
                action_decoder_state,
            ) = carry

            rollout_result_batch = (
                rollout_result_batch[0],
                rollout_result_batch[1][:, :-1],
            )

            rng, key = jax.random.split(key)
            (
                state_encoder_state,
                action_encoder_state,
                transition_model_state,
                state_decoder_state,
                action_decoder_state,
                loss_infos,
            ) = jax.jit(train_step)(
                rng,
                state_encoder_state,
                action_encoder_state,
                state_decoder_state,
                action_decoder_state,
                transition_model_state,
                rollout_result_batch,
                action_bounds,
                dt,
                loss_weights,
            )

            msg = None

            jax.experimental.host_callback.id_tap(
                lambda chunk, _: print(f"Chunk {chunk}"), chunk_i
            )

            jax.experimental.host_callback.id_tap(
                dump_to_wandb_for_tap, (loss_infos, chunk_i)
            )

            return (
                key,
                chunk_i + 1,
                state_encoder_state,
                action_encoder_state,
                transition_model_state,
                state_decoder_state,
                action_decoder_state,
            ), (msg, loss_infos)

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
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
        )

        (
            _,
            _,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
        ), (msg, loss_infos) = jax.lax.scan(
            process_batch, init, rollout_results_batched
        )

        return (
            key,
            epoch + 1,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
        ), loss_infos

        # jax.profiler.save_device_memory_profile("memory.prof")

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, epochs)

    rng, key = jax.random.split(key)
    init = (
        rng,
        0,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
    )

    jax.experimental.host_callback.id_tap(
        lambda rollout, _: print(f"Rollout {rollout}"), rollout_i
    )

    (
        _,
        _,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
    ), infos = jax.lax.scan(do_epoch, init, None, length=epochs)

    # jax.experimental.host_callback.id_tap(dump_infos_for_tap, (infos, rollout_i))

    return (
        key,
        rollout_i + 1,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
    ), _


rng, key = jax.random.split(key)
init = (
    rng,
    jnp.array(0),
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    state_decoder_state,
    action_decoder_state,
)

(
    _,
    _,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    state_decoder_state,
    action_decoder_state,
), _ = jax.lax.scan(do_rollout, init, None, length=rollouts)


jax.debug.print("Done!")

for i in range(16):
    ani = animate(rollout_result[0][i, ..., :7], shape_config=shape_config, dt=dt)
    plt.show()

pass
