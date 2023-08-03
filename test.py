import jax

import jax.numpy as jnp

from mass_matrix import mass_matrix
from bias_forces import bias_forces


step = 1

counter = 0


def cond(_):
    return counter < 10


def func(count):
    counter += step
    return None


final_count = jax.lax.while_loop(cond, func, None)

print(f"Whoop: {final_count}")
