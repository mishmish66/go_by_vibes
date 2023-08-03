import jax

import jax.numpy as jnp

from mass_matrix import mass_matrix
from bias_forces import bias_forces

from positions import make_positions

from contact_solver import iterative_solver

import timeit

from visualize import animate

from einops import einops, einsum

from simulate import physics_step

import matplotlib.pyplot as plt

# Generate random key
key = jax.random.PRNGKey(0)
rng, key = jax.random.split(key)

mass_matrix = jax.jit(mass_matrix)
bias_forces = jax.jit(bias_forces)

target = jnp.array([0, 0, 0, 0.5, -0.5, -0.5, 0.5], dtype=jnp.float32)
kp = jnp.array([0, 0, 0, 10.0, 10.0, 10.0, 10.0], dtype=jnp.float32)
kd = jnp.array([0, 0, 0, 0.1, 0.1, 0.1, 0.1], dtype=jnp.float32)

jax.random.PRNGKey(0)
envs = 4096
rng, key = jax.random.split(key)
q = (
    jnp.array([0, 0, -0.0, 0.5, -0.5, -0.5, 0.5], dtype=jnp.float32)
    + jax.random.normal(rng, (envs, 7)) * 0.01
)
rng, key = jax.random.split(key)
qd = (
    jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    + jax.random.normal(rng, (envs, 7)) * 0.001
)

dt = 0.02
substep = 10
total_time = 5.0

sim_dt = dt / substep
vmap_physics_step = jax.vmap(physics_step, (0, 0, None, None))


def control(q, qd):
    return (target - q) * kp - qd * kd


def scanf(carry, _):
    q, qd = carry
    q, qd = vmap_physics_step(q, qd)
    return (q, qd), q


_, qs = jax.lax.scan(scanf, (q, qd), None, length=total_time / sim_dt)

qs_step = qs[::substep, 0, :]

ani = animate(qs_step, shape_config=shape_config, dt=dt)

plt.show()
pass

# print("Starting")
# start_time = timeit.default_timer()
# m = mass_matrix_vmap(q, q_dot, mass_config, shape_config)
# end_time = timeit.default_timer()
# elapsed = end_time - start_time
# print("Elapsed time: ", elapsed)

# print("Starting")
# start_time = timeit.default_timer()
# of = other_forces_vmap(q, q_dot, mass_config, shape_config, 9.81)
# end_time = timeit.default_timer()
# elapsed = end_time - start_time
# print("Elapsed time: ", elapsed)
