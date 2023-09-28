from typing import Any

import jax
from jax import numpy as jnp

import flax
from flax import linen as nn

from einops import rearrange

encoded_state_dim = 32
encoded_action_dim = 16


class StateEncoder(nn.Module):
    @nn.compact
    def __call__(self, x) -> Any:
        x = nn.Dense(128, name="FC1")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC2")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC3")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC4")(x)
        x = nn.relu(x)
        x = nn.Dense(encoded_state_dim * 2, name="FC5")(x)
        x_mean = x[..., :encoded_state_dim]
        x_std = x[..., encoded_state_dim:]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class StateDecoder(nn.Module):
    @nn.compact
    def __call__(self, x) -> Any:
        x = nn.Dense(128, name="FC1")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC2")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC3")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC4")(x)
        x = nn.relu(x)
        x = nn.Dense(28, name="FC5")(x)
        x_mean = x[..., :14]
        x_std = x[..., 14:]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class ActionEncoder(nn.Module):
    @nn.compact
    def __call__(self, latent_action, latent_state) -> Any:
        x = jnp.concatenate([latent_action, latent_state], axis=-1)
        x = nn.Dense(128, name="FC1")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC2")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC3")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC4")(x)
        x = nn.relu(x)
        x = nn.Dense(encoded_action_dim * 2, name="FC5")(x)
        x_mean = x[..., :encoded_action_dim]
        x_std = x[..., encoded_action_dim:]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class ActionDecoder(nn.Module):
    @nn.compact
    def __call__(self, latent_action, latent_state) -> Any:
        x = jnp.concatenate([latent_action, latent_state], axis=-1)
        x = nn.Dense(128, name="FC1")(x)
        x = nn.relu(x)
        x = nn.Dense(128, name="FC2")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC3")(x)
        x = nn.relu(x)
        x = nn.Dense(64, name="FC4")(x)
        x = nn.relu(x)
        x = nn.Dense(8, name="FC5")(x)
        x_mean = x[..., :4]
        x_std = x[..., 4:]
        x_std = nn.softplus(x_std)
        x = jnp.concatenate([x_mean, x_std], axis=-1)
        return x


class TemporalEncoder(nn.Module):
    def setup(self, n=1e4):
        self.n = n

    def __call__(self, x, time) -> Any:
        d = x.shape[-1]
        indices = jnp.arange(d)
        denominators = jnp.power(self.n, indices / d)
        operands = time / denominators
        sins = jnp.sin(operands[0::2])
        cosines = jnp.cos(operands[1::2])

        freq_result = rearrange([sins, cosines], "f e -> (e f)")

        return x + freq_result


class TransformerLayer(nn.Module):
    def setup(self, dim, heads=8, dropout=0.1):
        self.attention = nn.MultiHeadDotProductAttention(
            num_heads=heads,
            out_features=dim,
            name="ATTN",
            dropout_rate=dropout,
        )

        self.mlp_up = nn.Dense(dim * 4, name="MLPU")
        self.mlp_down = nn.Dense(dim, name="MLPD")

    def __call__(self, queries, keys_values, mask=None):
        x = queries
        x = x + self.attention(queries, keys_values, mask)
        u = self.mlp_up(x)
        z = nn.relu(u)
        r = self.mlp_down(z)
        x = x + nn.relu(r)

        return x


class TransitionModel(nn.Module):
    def setup(
        self,
        n=1e4,
        latent_dim=512,
    ):
        self.temporal_encoder = TemporalEncoder(n=n)

        self.action_t_layers = {
            i: {
                "CA": TransformerLayer(
                    dim=latent_dim,
                    name=f"ACT_CA{i}",
                ),
                "SA": TransformerLayer(
                    dim=latent_dim,
                    name=f"ACT_SA{i}",
                ),
            }
            for i in range(2)
        }

        self.state_t_layers = {
            i: {
                "CA": TransformerLayer(
                    dim=latent_dim,
                    name=f"STA_CA{i}",
                ),
                "SA": TransformerLayer(
                    dim=latent_dim,
                    name=f"STA_SA{i}",
                ),
            }
            for i in range(6)
        }

        self.state_condenser = nn.Dense(encoded_state_dim * 2, name="SC")

    def __call__(
        self,
        states,
        actions,
        state_times,
        action_times,
        mask,
    ) -> Any:
        states = states[mask]
        # Upscale actions and states to latent dim
        states = self.state_expander(states)
        actions = self.action_expander(actions)

        # Apply temporal encodings
        states = self.temporal_encoder(states, state_times)
        actions = self.temporal_encoder(actions, action_times)

        # Apply transformer layers
        t_layer_indices = [*self.action_t_layers.keys(), *self.state_t_layers.keys()]
        for i in max(t_layer_indices):
            if i in self.state_t_layers:
                states = self.state_t_layers[i]["CA"](states, actions)
                states = self.state_t_layers[i]["SA"](states, states)

            if i in self.action_t_layers:
                actions = self.action_t_layers[i]["CA"](actions, states)
                actions = self.action_t_layers[i]["SA"](actions, actions)

        # Rescale states to original dim
        self.state_condenser(states)
        x_mean = states[..., :encoded_state_dim]
        x_std = nn.softplus(states[..., encoded_state_dim:])

        x = jnp.concatenate([x_mean, x_std], axis=-1)

        return x
