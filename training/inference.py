import jax
from jax import numpy as jnp

from .vibe_state import VibeState, TrainConfig

from .nets import (
    encoded_state_dim,
    encoded_action_dim,
    StateEncoder,
    ActionEncoder,
    TransitionModel,
    StateDecoder,
    ActionDecoder,
)


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


def get_latent_state_gaussian(
    state,
    vibe_state: VibeState,
    vibe_config: TrainConfig,
):
    latent_state_gaussian = vibe_config.state_encoder.apply(
        vibe_state.state_encoder_params,
        state,
    )
    return latent_state_gaussian


def encode_state(
    key,
    state,
    vibe_state: VibeState,
    vibe_config: TrainConfig,
):
    rng, key = jax.random.split(key)

    latent_state_gaussian = get_latent_state_gaussian(
        state,
        vibe_state,
        vibe_config,
    )
    latent_state = sample_gaussian(rng, latent_state_gaussian)

    return latent_state


def get_latent_action_gaussian(
    action,
    latent_state,
    vibe_state: VibeState,
    vibe_config: TrainConfig,
):
    latent_action_gaussian = vibe_config.action_encoder.apply(
        vibe_state.action_encoder_params,
        action,
        latent_state,
    )
    return latent_action_gaussian


def encode_action(
    key,
    action,
    latent_state,
    vibe_state: VibeState,
    vibe_config: TrainConfig,
):
    latent_action_gaussian = get_latent_action_gaussian(
        action,
        latent_state,
        vibe_state,
        vibe_config,
    )

    rng, key = jax.random.split(key)
    latent_action = sample_gaussian(rng, latent_action_gaussian)

    return latent_action


def get_state_space_gaussian(
    latent_state,
    vibe_state: VibeState,
    vibe_config: TrainConfig,
):
    state_gaussian = vibe_config.state_decoder.apply(
        vibe_state.state_decoder_params,
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
    latent_state,
    vibe_state: VibeState,
    vibe_config: TrainConfig,
):
    state_space_gaussian = get_state_space_gaussian(
        latent_state,
        vibe_state,
        vibe_config,
    )

    rng, key = jax.random.split(key)
    state = sample_gaussian(rng, state_space_gaussian)

    return state


def get_action_space_gaussian(
    latent_action,
    latent_state,
    vibe_state: VibeState,
    vibe_config: TrainConfig,
):
    action_gaussian = vibe_config.action_decoder.apply(
        vibe_state.action_decoder_params,
        latent_action,
        latent_state,
    )

    # Clamp the variance to at least 1e-6
    clamped_variance = action_gaussian[..., encoded_action_dim:] + 1e-6
    action_gaussian = jnp.concatenate(
        [action_gaussian[..., :encoded_action_dim], clamped_variance], axis=-1
    )

    return action_gaussian


def decode_action(
    key,
    latent_action,
    latent_state,
    vibe_state: VibeState,
    vibe_config: TrainConfig,
):
    action_space_gaussian = get_action_space_gaussian(
        latent_action,
        latent_state,
        vibe_state,
        vibe_config,
    )

    rng, key = jax.random.split(key)
    action = sample_gaussian(rng, action_space_gaussian)

    return action


def get_latent_state_prime_gaussians(
    latent_states,
    latent_actions,
    dt,
    vibe_state: VibeState,
    vibe_config: TrainConfig,
    mask=None,
):
    next_state_gaussian = vibe_config.transition_model.apply(
        vibe_state.transition_model_params,
        latent_states,
        latent_actions,
        jnp.arange(latent_actions.shape[0]) * dt,
        mask,
    )

    # Clamp the variance to at least 1e-6
    clamped_variance = next_state_gaussian[..., encoded_state_dim:] + 1e-6
    latent_state_prime_gaussians = jnp.concatenate(
        [next_state_gaussian[..., :encoded_state_dim], clamped_variance], axis=-1
    )

    return latent_state_prime_gaussians


def infer_states(
    key,
    latent_states,
    latent_actions,
    dt,
    vibe_state: VibeState,
    vibe_config: TrainConfig,
):
    rng, key = jax.random.split(key)

    latent_state_prime_gaussians = get_latent_state_prime_gaussians(
        latent_states,
        latent_actions,
        dt,
        vibe_state,
        vibe_config,
    )

    inferred_states = jax.vmap(sample_gaussian, (0, 0))(
        rng, latent_state_prime_gaussians
    )
    return inferred_states


def make_mask(mask_len, start_mask):
    mask = jnp.arange(mask_len)
    mask = mask < start_mask
    return mask