from physics.gen.mass_matrix import mass_matrix
from physics.gen.bias_forces import bias_forces

from physics.simulate import physics_step
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
from nets import StateEncoder, ActionEncoder, TransitionModel, encoded_state_dim, encoded_action_dim

import timeit

# Generate random key
key = jax.random.PRNGKey(0)

### Set up physics sim stuff
rng, key = jax.random.split(key)

mass_matrix = jax.jit(mass_matrix)
bias_forces = jax.jit(bias_forces)

mass_config = jnp.array([1.0, 0.25, 0.25, 0.04, 0.01, 0.01])
shape_config = jnp.array([1.0, 0.25, 0.25])

envs = 8192

rng, key = jax.random.split(key)
q = (
    jnp.array([0, 0, -0.0, 0.5, -0.5, -0.5, 0.5], dtype=jnp.float32)
    + jax.random.normal(rng, (envs, 7)) * 0.25
)
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

state_encoder = StateEncoder()
action_encoder = ActionEncoder()
transition_model = TransitionModel()
rng, key = jax.random.split(key)

params = state_encoder.init(rng, q)
z_state = state_encoder.apply(params, q)



qs = qs_sub[::substep, 0, :]

embeds = embedding_layer(qs)

ani = animate(qs, shape_config=shape_config, dt=dt)

plt.show()
pass
