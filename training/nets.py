from typing import Any

import jax
from jax import numpy as jnp

import flax
from flax import linen as nn

from einops import einsum, rearrange

encoded_state_dim = 3
encoded_action_dim = 2


class FreqLayer(nn.Module):
    out_dim: jax.Array

    def setup(self):
        pass

    def __call__(self, x) -> Any:
        d = x.shape[-1]
        per_dim = (((self.out_dim // d) - 1) // 2) + 1
        indices = jnp.arange(per_dim)
        freq_factor = 5 / jnp.power(1e4, 2 * indices / d)
        operands = einsum(x, freq_factor, "d, w -> w d")
        sins = jnp.sin(operands)
        cosines = jnp.cos(operands)

        freq_result = rearrange([sins, cosines], "f w d -> (d f w)")
        sliced_freq_result = freq_result[: self.out_dim - d]

        cat_result = jnp.concatenate([x, sliced_freq_result], axis=-1)

        return cat_result


class StateEncoder(nn.Module):
    def setup(self):
        self.freq_layer = FreqLayer(out_dim=64)

        self.dense_layers = [
            nn.Dense(dim, name=f"FC{i}")
            for i, dim in enumerate(
                [
                    64,
                    32,
                    32,
                    32,
                    encoded_state_dim * 2,
                ]
            )
        ]

    def __call__(self, x) -> Any:
        x = self.freq_layer(x)

        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)

        x = self.dense_layers[-1](x)
        x_mean = x[..., :encoded_state_dim]
        x_std = x[..., encoded_state_dim:]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class StateDecoder(nn.Module):
    state_dim: any

    def setup(self):
        self.dense_layers = [
            nn.Dense(d, name=f"FC{i}")
            for i, d in enumerate(
                [
                    128,
                    64,
                    64,
                    self.state_dim * 2,
                ]
            )
        ]

    def __call__(self, x) -> Any:
        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        x = self.dense_layers[-1](x)
        x_mean = x[..., : self.state_dim]
        x_std = x[..., self.state_dim :]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class ActionEncoder(nn.Module):
    def setup(self):
        self.freq_layer = FreqLayer(out_dim=64)

        self.dense_layers = [
            nn.Dense(dim, name=f"FC{i}")
            for i, dim in enumerate(
                [
                    64,
                    64,
                    32,
                    32,
                    encoded_action_dim * 2,
                ]
            )
        ]

    def __call__(self, action, latent_state) -> Any:
        freq_action = self.freq_layer(action)
        x = jnp.concatenate([freq_action, latent_state], axis=-1)

        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)

        x = self.dense_layers[-1](x)

        x_mean = x[..., :encoded_action_dim]
        x_std = x[..., encoded_action_dim:]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class ActionDecoder(nn.Module):
    act_dim: any

    def setup(self):
        self.dense_layers = [
            nn.Dense(d, name=f"FC{i}")
            for i, d in enumerate(
                [
                    128,
                    128,
                    64,
                    64,
                    8,
                    self.act_dim * 2,
                ]
            )
        ]

    def __call__(self, latent_action, latent_state) -> Any:
        x = jnp.concatenate([latent_action, latent_state], axis=-1)
        for layer in self.dense_layers[:-1]:
            x = layer(x)
            x = nn.relu(x)
        x = self.dense_layers[-1](x)
        x_mean = x[..., : self.act_dim]
        x_std = x[..., self.act_dim :]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class TemporalEncoder(nn.Module):
    n: float

    def setup(self):
        pass

    def __call__(self, x, time) -> Any:
        d = x.shape[-1]
        indices = jnp.arange(d)
        denominators = jnp.power(self.n, indices / d)
        operands = x / denominators
        sins = jnp.sin(operands[0::2])
        cosines = jnp.cos(operands[1::2])

        # interleave
        freq_result = jnp.empty_like(operands)
        freq_result = freq_result.at[0::2].set(sins)
        freq_result = freq_result.at[1::2].set(cosines)

        return x + freq_result


class TransformerLayer(nn.Module):
    dim: int
    heads: int
    dropout: float

    def setup(self):
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=self.heads,
            out_features=self.dim,
            name="ATTN",
            dropout_rate=self.dropout,
        )

        self.mlp_up = nn.Dense(self.dim * 4, name="MLPU")
        self.mlp_down = nn.Dense(self.dim, name="MLPD")

    def __call__(self, queries, keys_values, mask=None):
        x = queries
        x = x + self.attention(queries, keys_values, mask)
        u = self.mlp_up(x)
        z = nn.relu(u)
        r = self.mlp_down(z)
        x = x + nn.relu(r)

        return x


def make_inds(mask_len, first_known_i):
    inds = jnp.arange(mask_len) - first_known_i
    return inds


class TransitionModel(nn.Module):
    encoder_n: float
    n_layers: int
    latent_dim: int
    heads: int

    def setup(self):
        self.temporal_encoder = TemporalEncoder(n=self.encoder_n)

        self.state_action_expander = nn.Dense(self.latent_dim, name="ACTION_EXPANDER")

        self.t_layers = [
            TransformerLayer(
                dim=self.latent_dim,
                heads=self.heads,
                dropout=0.0,
                name=f"ATTN_{i}",
            )
            for i in range(self.n_layers)
        ]

        self.state_condenser = nn.Dense(encoded_state_dim * 2, name="STATE_CONDENSER")

    def __call__(
        self,
        initial_latent_state,
        latent_actions,
        times,
        first_known_action_i,
    ) -> Any:
        inds = make_inds(latent_actions.shape[0], first_known_action_i)
        mask_time_inds = einsum(inds, inds < 0, "i, i->i")

        # Apply temporal encodings
        latent_actions_temp = jax.vmap(self.temporal_encoder, (0, 0))(
            latent_actions, times[mask_time_inds]
        )

        state_actions = jax.vmap(
            lambda s, a: jnp.concatenate([s, a]),
            (None, 0),
        )(initial_latent_state, latent_actions_temp)

        # Upscale actions and state to latent dim
        x = jax.vmap(self.state_action_expander.__call__)(state_actions)

        mask = inds >= 0

        # Apply transformer layers
        for t_layer in self.t_layers:
            x = t_layer(x, x, mask)

        # Rescale states to original dim
        x = self.state_condenser(x)
        latent_state_prime_mean = x[..., :encoded_state_dim]
        latent_state_prime_std = nn.softplus(x[..., encoded_state_dim:])

        latent_state_prime_gauss_params = jnp.concatenate(
            [
                latent_state_prime_mean,
                latent_state_prime_std,
            ],
            axis=-1,
        )

        return latent_state_prime_gauss_params
