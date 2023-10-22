from finger import Finger

from jax import numpy as jnp
import jax
from jax.experimental.host_callback import id_tap, barrier_wait

import imageio

state = Finger.init()

action = jnp.ones(2)

from jax import config
config.update("jax_disable_jit", True)

def save_video_for_tap(states, _):
    fps = 24
    if len(states.shape) > 2:
        save_video_for_tap(states[0], _)
        return
    else:
        frames = Finger.host_make_video(states, Finger.get_config(), fps)
        imageio.mimwrite("test.mp4", frames, fps=fps)

def do_thing(key):
    
    def scanf(carry, key):
        state, i = carry
        
        rng, key = jax.random.split(key)
        action = jax.random.uniform(rng, shape=(2,))
        
        next_state = Finger.step(state, action, None)
        
        return (next_state, i + 1), next_state

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, 512)
    _, states = jax.lax.scan(scanf, (state, 0), rngs)

    id_tap(save_video_for_tap, states)
    
    return jnp.linalg.norm(states, ord=1, axis=-1).mean(), states

key = jax.random.PRNGKey(0)
_, states = jax.vmap(do_thing)(jax.random.split(key, 1))

# id_tap(save_video_for_tap, states[0])

pass