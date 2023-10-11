from typing import Any

import jax
from jax import numpy as jnp

import flax
from flax import linen as nn

from einops import einsum, rearrange

encoded_state_dim = 16
encoded_action_dim = 16


class FreqLayer(nn.Module):
    out_dim: jax.Array

    def setup(self):
        pass

    def __call__(self, x) -> Any:
        d = x.shape[-1]
        per_dim = (((self.out_dim // d) + 1) // 2) + 1
        indices = jnp.arange(per_dim)
        freq_factor = 5 / jnp.power(1e4, 2 * indices / d)
        operands = einsum(x, freq_factor, "d, w -> w d")
        sins = jnp.sin(operands)
        cosines = jnp.cos(operands)

        freq_result = rearrange([sins, cosines], "f w d -> (d f w)")
        sliced_freq_result = freq_result[: self.out_dim]

        return sliced_freq_result


class StateEncoder(nn.Module):
    def setup(self):
        self.freq_layer = FreqLayer(out_dim=256)

        self.dense_layers = [
            nn.Dense(dim, name=f"FC{i}")
            for i, dim in enumerate(
                [
                    128,
                    128,
                    128,
                    64,
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
        self.freq_layer = FreqLayer(out_dim=256)

        self.dense_layers = [
            nn.Dense(dim, name=f"FC{i}")
            for i, dim in enumerate(
                [
                    128,
                    128,
                    128,
                    64,
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

        freq_result = rearrange([sins, cosines], "f e -> (e f)")

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


class TransitionModel(nn.Module):
    n: float
    latent_dim: int
    heads: int

    def setup(self):
        self.temporal_encoder = TemporalEncoder(n=self.n)

        self.state_expander = nn.Dense(self.latent_dim, name="SE")
        self.action_expander = nn.Dense(self.latent_dim, name="AE")

        self.action_t_layers = {
            i: {
                "CA": TransformerLayer(
                    dim=self.latent_dim,
                    heads=self.heads,
                    dropout=0.0,
                    name=f"ACT_CA{i}",
                ),
                "SA": TransformerLayer(
                    dim=self.latent_dim,
                    heads=self.heads,
                    dropout=0.0,
                    name=f"ACT_SA{i}",
                ),
            }
            for i in range(2)
        }

        self.state_t_layers = {
            i: {
                "CA": TransformerLayer(
                    dim=self.latent_dim,
                    heads=self.heads,
                    dropout=0.0,
                    name=f"STA_CA{i}",
                ),
                "SA": TransformerLayer(
                    dim=self.latent_dim,
                    heads=self.heads,
                    dropout=0.0,
                    name=f"STA_SA{i}",
                ),
            }
            for i in range(6)
        }

        self.state_condenser = nn.Dense(encoded_state_dim * 2, name="SC")

    def __call__(
        self,
        latent_states,
        latent_actions,
        times,
        mask,
    ) -> Any:
        # Upscale actions and states to latent dim
        states = self.state_expander(latent_states)
        actions = self.action_expander(latent_actions)

        # Apply temporal encodings
        states = jax.vmap(self.temporal_encoder, (0, 0))(
            states, times[: states.shape[0]]
        )
        actions = jax.vmap(self.temporal_encoder, (0, 0))(actions, times)

        # Apply transformer layers
        t_layer_indices = [*self.action_t_layers.keys(), *self.state_t_layers.keys()]
        for i in range(max(t_layer_indices)):
            if i in self.state_t_layers:
                states = self.state_t_layers[i]["CA"](states, actions, mask)
                states = self.state_t_layers[i]["SA"](states, states, mask)

            if i in self.action_t_layers:
                actions = self.action_t_layers[i]["CA"](actions, states)
                actions = self.action_t_layers[i]["SA"](actions, actions)

        # Rescale states to original dim
        latent_states_prime = self.state_condenser(states)
        x_mean = latent_states_prime[..., :encoded_state_dim]
        x_std = nn.softplus(latent_states_prime[..., encoded_state_dim:])

        x = jnp.concatenate([x_mean, x_std], axis=-1)

        return x
