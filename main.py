from physics.gen.mass_matrix import mass_matrix
from physics.gen.bias_forces import bias_forces

from physics.simulate import step
from physics.visualize import animate

import jax
import jax.numpy as jnp
from clu import metrics
import flax
from flax import linen as nn
from flax import struct
import optax

from einops import einops, einsum
import matplotlib.pyplot as plt

from embeds import EmbeddingLayer
from rollout import collect_rollout
from training import create_train_state, train_step
from nets import (
    StateEncoder,
    ActionEncoder,
    TransitionModel,
    StateDecoder,
    ActionDecoder,
    encoded_state_dim,
    encoded_action_dim,
)
from policy import stupid_policy

import timeit

# Generate random key
key = jax.random.PRNGKey(1)

### Set up physics sim stuff
rng, key = jax.random.split(key)

mass_matrix = jax.jit(mass_matrix)
bias_forces = jax.jit(bias_forces)

mass_config = jnp.array([1.0, 0.25, 0.25, 0.04, 0.01, 0.01])
shape_config = jnp.array([1.0, 0.25, 0.25])

envs = 16

rng, key = jax.random.split(key)
home_q = jnp.array([0, 0, -0.0, 0.5, -0.5, -0.5, 0.5], dtype=jnp.float32)
# home_q = jnp.array([0, 0, -0.0, -jnp.pi, -0.0, -jnp.pi, 0.0], dtype=jnp.float32)
start_q = home_q + jax.random.normal(rng, (envs, 7)) * 0.25

rng, key = jax.random.split(key)
qd = (
    jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    + jax.random.normal(rng, (envs, 7)) * 0.1
)

dt = 0.02
substep = 2
total_time = 5.0


### Set up RL stuff

loops = 4096

state_embedding_layer = EmbeddingLayer(256)
action_embedding_layer = EmbeddingLayer(128)

# rng, key = jax.random.split(key)

# params = state_encoder.init(rng, q)
# z_state = state_encoder.apply(params, q)

state_embeddings = EmbeddingLayer(encoded_state_dim)
action_embeddings = EmbeddingLayer(encoded_action_dim)

state_encoder = StateEncoder()
action_encoder = ActionEncoder()
transition_model = TransitionModel()
state_decoder = StateDecoder()
action_decoder = ActionDecoder()

rng, key = jax.random.split(key)
state_encoder_state = create_train_state(state_encoder, 14, rng, learning_rate=0.001)

rng, key = jax.random.split(key)
action_encoder_state = create_train_state(action_encoder, 4, rng, learning_rate=0.001)

rng, key = jax.random.split(key)
transition_model_state = create_train_state(
    transition_model, encoded_state_dim + encoded_action_dim, rng, learning_rate=0.001
)

rng, key = jax.random.split(key)
state_decoder_state = create_train_state(
    state_decoder, encoded_state_dim, rng, learning_rate=0.001
)

rng, key = jax.random.split(key)
action_decoder_state = create_train_state(
    action_decoder, encoded_action_dim, rng, learning_rate=0.001
)


def policy(q, qd, key):
    state = jnp.concatenate([q, qd], axis=-1)
    target_state = jnp.concatenate([home_q, jnp.zeros_like(home_q)], axis=-1)

    action = stupid_policy(
        state_encoder_state,
        action_decoder_state,
        transition_model_state,
        state,
        target_state,
        32,
        key,
        mass_config,
        shape_config,
        dt,
        refine_steps=4,
    )

    clipped_action = jnp.clip(action, -5.0, 5.0)

    return clipped_action


policy = jax.tree_util.Partial(policy)


rng, key = jax.random.split(key)
rngs = jax.random.split(rng, start_q.shape[:-1])

rollout_result = None

train_step(
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    transition_model_state,
    start_q,
    qd,
    mass_config,
    shape_config,
    policy,
    dt,
    substep,
    int(total_time / dt),
    key,
    envs
)

# jit_collect = jax.jit(collect_rollout, static_argnames=("substep", "steps"))
# rollout_result = jax.vmap(jit_collect, in_axes=((0, 0) + (None,) * 6 + (0,)))(
#     start_q,
#     qd,
#     mass_config,
#     shape_config,
#     policy,
#     dt,
#     substep,
#     int(total_time / dt),
#     rngs,
# )

jax.debug.print("Done!")

for i in range(16):
    ani = animate(rollout_result[0][i, ..., :7], shape_config=shape_config, dt=dt)
    plt.show()

pass
