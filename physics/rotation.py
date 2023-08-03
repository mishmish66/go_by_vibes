from jax import numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax

def rotmat(theta):
    return jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ]
    )
