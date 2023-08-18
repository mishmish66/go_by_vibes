import jax
from jax import numpy as jnp

from einops import einsum, rearrange


class EmbeddingLayer:
    def __init__(self, embed_dim=256):
        self.embed_dim = embed_dim

    def __call__(self, x):
        embed_factor = self.embed_dim / x.shape[-1]
        embed_factor = jnp.ceil(embed_factor)
        freqs = jnp.arange(0, embed_factor / 2, 1)

        trig_in = einsum(x, freqs, "i, j -> i j")

        sins = jnp.sin(trig_in)
        coss = jnp.cos(trig_in)

        x = rearrange([sins, coss], "t d f -> (f d t)")
        x = x[:self.embed_dim]
        
        return x
