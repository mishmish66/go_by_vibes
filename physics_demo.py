from physics.gen.mass_matrix import mass_matrix
from physics.gen.bias_forces import bias_forces

from physics.simulate import step
from physics.visualize import animate

import jax
import jax.numpy as jnp
from einops import einops, einsum
import matplotlib.pyplot as plt

import timeit

# Generate random key
key = jax.random.PRNGKey(0)

### Set up physics sim stuff
rng, key = jax.random.split(key)

mass_matrix = jax.jit(mass_matrix)
bias_forces = jax.jit(bias_forces)

mass_config = jnp.array([1.0, 0.25, 0.25, 0.04, 0.01, 0.01])
shape_config = jnp.array([1.0, 0.25, 0.25])

# target = jnp.array([0, 0, 0, 0.5, -0.5, -0.5, 0.5], dtype=jnp.float32)
target = jnp.array([0, 0, -0.0, -jnp.pi, -0.0, -jnp.pi, 0.0], dtype=jnp.float32)
kp = jnp.array([0, 0, 0, 10.0, 10.0, 10.0, 10.0], dtype=jnp.float32)
kd = jnp.array([0, 0, 0, 0.1, 0.1, 0.1, 0.1], dtype=jnp.float32)

jax.random.PRNGKey(0)
envs = 8
rng, key = jax.random.split(key)

q = target + jax.random.normal(rng, (envs, 7)) * 0.01
rng, key = jax.random.split(key)
qd = (
    jnp.array([0, 0, 0, 0, 0, 0, 0], dtype=jnp.float32)
    + jax.random.normal(rng, (envs, 7)) * 0.001
)

dt = 0.02
substep = 10
total_time = 5.0

sim_dt = dt / substep
vmap_physics_step = jax.vmap(step, (0, 0, None, None, 0, None))


def control(q, qd):
    return (q - target) * kp + qd * kd


def scanf(carry, _):
    q, qd = carry
    control_t = control(q, qd)
    q, qd = vmap_physics_step(q, qd, mass_config, shape_config, control_t, sim_dt)
    return (q, qd), q

_, qs = jax.lax.scan(scanf, (q, qd), None, length=total_time / sim_dt)

import time

start = time.time()
_, qs = jax.lax.scan(scanf, (q, qd), None, length=total_time / sim_dt)
end = time.time()
print(end - start)


qs_step = qs[::substep, 0, :]

ani = animate(qs_step, shape_config=shape_config, dt=dt)

plt.show()
