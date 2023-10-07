from typing import Any

import jax
from jax import numpy as jnp

import flax
from flax import linen as nn

from einops import einsum, rearrange

encoded_state_dim = 32
encoded_action_dim = 16


def sample_gaussian(key, gaussian):
    dim = gaussian.shape[-1] // 2

    old_pre_shape = gaussian.shape[:-1]

    flat_gaussians = jnp.reshape(gaussian, (-1, gaussian.shape[-1]))

    flat_mean = flat_gaussians[:, :dim]
    flat_variance_vectors = flat_gaussians[:, dim:]

    rng, key = jax.random.split(key)
    normal = jax.random.normal(rng, flat_mean.shape)

    flat_result = flat_mean + normal * flat_variance_vectors

    result = jnp.reshape(flat_result, (*old_pre_shape, dim))

    return result


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
        self.freq_layer = FreqLayer(out_dim=128)

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
        x = jnp.concatenate([action, latent_state], axis=-1)

        x = self.freq_layer(x)

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

    def setup(self):
        self.temporal_encoder = TemporalEncoder(n=self.n)

        # temporal_encoder_params = self.temporal_encoder.init(
        #     jax.random.PRNGKey(0),
        #     jnp.zeros((1, self.latent_dim)),
        #     jnp.zeros([1, 1]),
        # )

        self.state_expander = nn.Dense(self.latent_dim, name="SE")
        self.action_expander = nn.Dense(self.latent_dim, name="AE")

        self.action_t_layers = {
            i: {
                "CA": TransformerLayer(
                    dim=self.latent_dim,
                    heads=8,
                    dropout=0.0,
                    name=f"ACT_CA{i}",
                ),
                "SA": TransformerLayer(
                    dim=self.latent_dim,
                    heads=8,
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
                    heads=8,
                    dropout=0.0,
                    name=f"STA_CA{i}",
                ),
                "SA": TransformerLayer(
                    dim=self.latent_dim,
                    heads=8,
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
    ) -> Any:
        # Upscale actions and states to latent dim
        states = self.state_expander(latent_states)
        actions = self.action_expander(latent_actions)

        # Apply temporal encodings
        states = jax.vmap(self.temporal_encoder, (0, 0))(
            states, times[: states.shape[0]]
        )
        actions = jax.vmap(self.temporal_encoder, (0, 0))(actions, times)

        # jax.debug.print(
        #     "NaN before attn: {}", jnp.isnan(states).any() + jnp.isnan(actions).any()
        # )

        # Apply transformer layers
        t_layer_indices = [*self.action_t_layers.keys(), *self.state_t_layers.keys()]
        for i in range(max(t_layer_indices)):
            if i in self.state_t_layers:
                states = self.state_t_layers[i]["CA"](states, actions)
                states = self.state_t_layers[i]["SA"](states, states)

            if i in self.action_t_layers:
                actions = self.action_t_layers[i]["CA"](actions, states)
                actions = self.action_t_layers[i]["SA"](actions, actions)

        # Rescale states to original dim
        latent_states_prime = self.state_condenser(states)
        x_mean = latent_states_prime[..., :encoded_state_dim]
        x_std = nn.softplus(latent_states_prime[..., encoded_state_dim:])

        x = jnp.concatenate([x_mean, x_std], axis=-1)

        return x


def encode_state(
    key,
    state_encoder_state,
    state,
    state_encoder_params=None,
):
    # You were investigating the size of state here cuz it was wack somewhere
    if state_encoder_params is None:
        state_encoder_params = state_encoder_state.params

    rng, key = jax.random.split(key)
    latent_state_gaussian = state_encoder_state.apply_fn(
        {"params": state_encoder_params},
        state,
    )
    latent_state = sample_gaussian(rng, latent_state_gaussian)

    return latent_state


def encode_action(
    key,
    action_encoder_state,
    action,
    latent_state,
    action_encoder_params=None,
):
    if action_encoder_params is None:
        action_encoder_params = action_encoder_state.params

    rng, key = jax.random.split(key)
    latent_action_gaussian = action_encoder_state.apply_fn(
        {"params": action_encoder_params},
        action,
        latent_state,
    )
    latent_action = sample_gaussian(rng, latent_action_gaussian)

    return latent_action


def encode_state_action(
    key,
    action,
    state,
    state_encoder_state,
    action_encoder_state,
    state_encoder_params=None,
    action_encoder_params=None,
):
    if state_encoder_params is None:
        state_encoder_params = state_encoder_state.params
    if action_encoder_params is None:
        action_encoder_params = action_encoder_state.params

    rng, key = jax.random.split(key)
    latent_state = encode_state(
        key,
        state,
        state_encoder_state,
        state_encoder_params,
    )

    rng, key = jax.random.split(key)
    latent_action = encode_action(
        key,
        action,
        latent_state,
        action_encoder_state,
        action_encoder_params,
    )

    return latent_action, latent_state


def get_state_space_gaussian(
    state_decoder_state,
    latent_state,
    state_decoder_params=None,
):
    if state_decoder_params is None:
        state_decoder_params = state_decoder_state.params

    state_gaussian = state_decoder_state.apply_fn(
        {"params": state_decoder_params},
        latent_state,
    )

    # Clamp the variance to at least 1e-6
    clamped_variance = jnp.clip(state_gaussian[..., encoded_state_dim:], 1e-6, None)
    state_gaussian = jnp.concatenate(
        [state_gaussian[..., :encoded_state_dim], clamped_variance], axis=-1
    )

    return state_gaussian


def decode_state(
    key,
    state_decoder_state,
    latent_state,
    state_decoder_params=None,
):
    if state_decoder_params is None:
        state_decoder_params = state_decoder_state.params

    state_space_gaussian = get_state_space_gaussian(
        state_decoder_state,
        latent_state,
        state_decoder_params,
    )

    rng, key = jax.random.split(key)
    state = sample_gaussian(rng, state_space_gaussian)

    return state


def get_action_space_gaussian(
    action_decoder_state,
    latent_action,
    latent_state,
    action_decoder_params=None,
):
    if action_decoder_params is None:
        action_decoder_params = action_decoder_state.params

    action_gaussian = action_decoder_state.apply_fn(
        {"params": action_decoder_params},
        latent_action,
        latent_state,
    )

    # Clamp the variance to at least 1e-6
    clamped_variance = jnp.clip(action_gaussian[..., encoded_action_dim:], 1e-6, None)
    action_gaussian = jnp.concatenate(
        [action_gaussian[..., :encoded_action_dim], clamped_variance], axis=-1
    )

    return action_gaussian


def decode_action(
    key,
    action_decoder_state,
    latent_action,
    latent_state,
    action_decoder_params=None,
):
    if action_decoder_params is None:
        action_decoder_params = action_decoder_state.params

    action_space_gaussian = get_action_space_gaussian(
        action_decoder_state,
        latent_action,
        latent_state,
        action_decoder_params,
    )

    rng, key = jax.random.split(key)
    action = sample_gaussian(rng, action_space_gaussian)

    return action


def get_next_state_space_gaussians(
    transition_model_state,
    latent_states,
    latent_actions,
    dt,
    transition_model_params=None,
):
    if transition_model_params is None:
        transition_model_params = transition_model_state.params

    # jax.debug.print("latent_states: {}", latent_states)
    # jax.debug.print("latent_actions: {}", latent_actions)

    next_state_gaussian = transition_model_state.apply_fn(
        {"params": transition_model_params},
        latent_states,
        latent_actions,
        jnp.arange(latent_actions.shape[0]) * dt,
    )

    # jax.debug.print("next state gaussian: {}", next_state_gaussian)

    # Clamp the variance to at least 1e-6
    clamped_variance = next_state_gaussian[..., encoded_state_dim:] + 1e-6
    next_state_gaussian = jnp.concatenate(
        [next_state_gaussian[..., :encoded_state_dim], clamped_variance], axis=-1
    )

    return next_state_gaussian

def infer_states(
    key,
    transition_model_state,
    latent_states,
    latent_actions,
    dt,
    transition_model_params=None,
    return_gaussians=False,
):
    if transition_model_params is None:
        transition_model_params = transition_model_state.params

    rng, key = jax.random.split(key)

    inferred_state_gaussians = get_next_state_space_gaussians(
        transition_model_state,
        latent_states,
        latent_actions,
        dt,
        transition_model_params,
    )

    # jax.debug.print("NaN in inferred_state_gaussians: {}", jnp.isnan(inferred_state_gaussians).any())

    inferred_states = sample_gaussian(rng, inferred_state_gaussians)

    # jax.debug.print("NaN in inferred_states: {}", jnp.isnan(inferred_states).any())

    if return_gaussians:
        return inferred_states, inferred_state_gaussians
    else:
        return inferred_states
