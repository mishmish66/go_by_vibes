import jax
from jax import numpy as jnp
from jax.scipy.stats.multivariate_normal import pdf as multinorm_pdf
from jax.scipy.stats.multivariate_normal import logpdf as multinorm_logpdf
from jax.nn import sigmoid

from einops import einsum, rearrange

from nets import (
    encoded_state_dim,
    encoded_action_dim,
    infer_states,
    sample_gaussian,
    encode_state,
    encode_action,
    encode_state_action,
    get_state_space_gaussian,
    get_action_space_gaussian,
    get_next_state_space_gaussians,
)


def make_other_infos(
    key,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    state_decoder_state,
    action_decoder_state,
    states,
    actions,
    dt,
):
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[0])
    latent_states = jax.vmap(encode_state, in_axes=(0, None, 0))(
        rngs, state_encoder_state, states
    )

    # This next part is just for debugging and you should delete soon
    latent_state_gaussian_params = jax.vmap(state_encoder_state.apply_fn, (None, 0))(
        {"params": state_encoder_state.params}, states
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, actions.shape[0])
    latent_actions = jax.vmap(encode_action, in_axes=(0, None, 0, 0))(
        rngs, action_encoder_state, actions, latent_states[:-1]
    )

    rng, key = jax.random.split(key)
    latent_states_prime = infer_states(
        rng,
        transition_model_state,
        latent_states[:-1],
        latent_actions,
        dt,
    )

    state_space_gaussians = jax.vmap(get_state_space_gaussian, in_axes=(None, 0))(
        state_decoder_state, latent_states
    )
    action_space_gaussians = jax.vmap(get_action_space_gaussian, in_axes=(None, 0, 0))(
        action_decoder_state, latent_actions, latent_states[:-1]
    )
    state_space_gaussians_prime = jax.vmap(get_state_space_gaussian, in_axes=(None, 0))(
        state_decoder_state, latent_states_prime
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, state_space_gaussians.shape[0])
    reconstructed_states = jax.vmap(sample_gaussian, in_axes=(0, 0))(
        rngs, state_space_gaussians
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, action_space_gaussians.shape[0])
    reconstructed_actions = jax.vmap(sample_gaussian, in_axes=(0, 0))(
        rngs, action_space_gaussians
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, state_space_gaussians_prime.shape[0])
    reconstructed_states_prime = jax.vmap(sample_gaussian, in_axes=(0, 0))(
        rngs, state_space_gaussians_prime
    )

    state_reconstruction_diffs = reconstructed_states - states
    state_reconstruction_diff_mags = jnp.linalg.norm(
        state_reconstruction_diffs, axis=-1
    )

    reconstructed_states_var_mags = jnp.abs(reconstructed_states[14:])

    latent_state_gaussian_vars = latent_state_gaussian_params[..., encoded_state_dim:]

    next_latent_state_gaussian_params = get_next_state_space_gaussians(
        transition_model_state, latent_states[:-1, ...], latent_actions, dt
    )

    rng, key = jax.random.split(key)
    one_of_next_latent_state_gaussian_params = jax.random.choice(
        rng, next_latent_state_gaussian_params, axis=0
    )

    return {
        "forward": {
            "one_of_next_latent_state_gaussian_params": one_of_next_latent_state_gaussian_params,
            "mean_latent_state_prime_mag": jnp.mean(
                jnp.abs(latent_states_prime), axis=-1
            ),
            "mean_reconstructed_state_prime_mag": jnp.mean(
                jnp.abs(reconstructed_states_prime), axis=-1
            ),
        },
        "state": {
            "mean_state_reconstruction_diffs_mag": jnp.mean(
                state_reconstruction_diff_mags, axis=-1
            ),
            "med_state_rec_diff_mag": jnp.median(
                state_reconstruction_diff_mags, axis=-1
            ),
            "mean_state_mag": jnp.mean(jnp.abs(states), axis=-1),
            "mean_latent_state_mag": jnp.mean(jnp.abs(latent_states), axis=-1),
            "min_latent_state_gauss_var_mag": jnp.min(
                latent_state_gaussian_vars, axis=-1
            ),
            "reconstructed_states_var_mags": jnp.mean(
                reconstructed_states_var_mags, axis=-1
            ),
        },
        "action": {
            "mean_latent_action_mag": jnp.mean(
                jnp.abs(latent_actions[:encoded_action_dim]), axis=-1
            ),
            "mean_action_mag": jnp.mean(jnp.abs(actions), axis=-1),
            "mean_reconstructed_state_mag": jnp.mean(
                jnp.abs(reconstructed_states), axis=-1
            ),
            "mean_reconstructed_action_mag": jnp.mean(
                jnp.abs(reconstructed_actions), axis=-1
            ),
        },
    }
