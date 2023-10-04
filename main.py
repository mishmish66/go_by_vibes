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
total_time = 2.0


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

rng, key = jax.random.split(key)
state_encoder_state = create_train_state(state_encoder, [14], rng, learning_rate=0.001)

rng, key = jax.random.split(key)
action_encoder_state = create_train_state(
    action_encoder, [4, encoded_state_dim], rng, learning_rate=0.001
)

rng, key = jax.random.split(key)
transition_model_state = create_train_state(
    transition_model,
    [(16, encoded_state_dim), (16, encoded_action_dim), (16,)],
    rng,
    learning_rate=0.001,
)

rng, key = jax.random.split(key)
state_decoder_state = create_train_state(
    state_decoder, [encoded_state_dim], rng, learning_rate=0.005
)

rng, key = jax.random.split(key)
action_decoder_state = create_train_state(
    action_decoder, [encoded_action_dim, encoded_state_dim], rng, learning_rate=0.005
)

action_bounds = jnp.array([1.0, 1.0, 1.0, 1.0])


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

rollouts = 1024
trajectories_per_rollout = 1024
epochs = 1024
minibatch = 64

state_dict = {
    "state_encoder_state": state_encoder_state,
    "action_encoder_state": action_encoder_state,
    "transition_model_state": transition_model_state,
    "state_decoder_state": state_decoder_state,
    "action_decoder_state": action_decoder_state,
}

for rollout in range(rollouts):
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

    def do_epoch(epoch, key):
        # (
        #     epoch,
        #     state_encoder_state,
        #     action_encoder_state,
        #     transition_model_state,
        #     state_decoder_state,
        #     action_decoder_state,
        #     loss_infos,
        # ) = carry_pack

        print(f"Epoch: {epoch}")

        # shuffle the data
        rng, key = jax.random.split(key)
        # Make a shuffled list of indices
        indices = jax.random.permutation(rng, rollout_result[0].shape[0])
        states = rollout_result[0][indices]
        actions = rollout_result[1][indices]

        infos = []

        start_i = 0

        def process_batch(carry, rollout_result_batch):
            (
                key,
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
            )

            # rng, key = jax.random.split(key)
            # (
            #     state_encoder_state,
            #     action_encoder_state,
            #     transition_model_state,
            #     state_decoder_state,
            #     action_decoder_state,
            #     msg,
            # ) = compute_metrics(
            #     rng,
            #     state_encoder_state,
            #     action_encoder_state,
            #     state_decoder_state,
            #     action_decoder_state,
            #     transition_model_state,
            #     rollout_result_batch,
            #     action_bounds,
            #     dt,
            # )
            msg = None

            # info_msgs = make_info_msgs(loss_infos)
            
            return (
                key,
                state_encoder_state,
                action_encoder_state,
                transition_model_state,
                state_decoder_state,
                action_decoder_state,
            ), (msg, loss_infos)

        states_batched = einops.rearrange(states, "(b r) t d -> b r t d", b=minibatch)
        actions_batched = einops.rearrange(actions, "(b r) t d -> b r t d", b=minibatch)

        rollout_results_batched = (states_batched, actions_batched)

        rng, key = jax.random.split(key)
        init = (
            rng,
            state_dict["state_encoder_state"],
            state_dict["action_encoder_state"],
            state_dict["transition_model_state"],
            state_dict["state_decoder_state"],
            state_dict["action_decoder_state"],
        )
        
        # Do one batch as a test
        # (
        #     key,
        #     state_encoder_state,
        #     action_encoder_state,
        #     transition_model_state,
        #     state_decoder_state,
        #     action_decoder_state,
        # ), (msg, loss_infos) = process_batch(init, (rollout_results_batched[0][0], rollout_results_batched[1][0]))

        (
            key,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
        ), (msg, loss_infos) = jax.lax.scan(process_batch, init, rollout_results_batched)

        dump_infos("infos", loss_infos, epoch)

        state_dict["state_encoder_state"] = state_encoder_state
        state_dict["action_encoder_state"] = action_encoder_state
        state_dict["transition_model_state"] = transition_model_state
        state_dict["state_decoder_state"] = state_decoder_state
        state_dict["action_decoder_state"] = action_decoder_state

        # return (
        #     epoch + 1,
        #     state_encoder_state,
        #     action_encoder_state,
        #     transition_model_state,
        #     state_decoder_state,
        #     action_decoder_state,
        #     final_infos,
        # )

        # jax.profiler.save_device_memory_profile("memory.prof")

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, epochs)

    for i in range(epochs):
        do_epoch(i, rngs[i])


jax.debug.print("Done!")

for i in range(16):
    ani = animate(rollout_result[0][i, ..., :7], shape_config=shape_config, dt=dt)
    plt.show()

pass
