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
    latent_states_prime, latent_state_prime_gaussians = infer_states(
        rng,
        transition_model_state,
        latent_states[:-1],
        latent_actions,
        dt,
        None,
        True,
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
    state_reconstruction_diff_mags = jnp.abs(state_reconstruction_diffs)

    reconstructed_states_var_mags = jnp.abs(reconstructed_states[14:])

    latent_state_gaussian_vars = latent_state_gaussian_params[..., encoded_state_dim:]

    return {
        "forward": {
            "latent_state_prime_mags": jnp.linalg.norm(latent_states_prime, axis=-1),
            "latent_state_prime_gauss_var_mags": latent_state_prime_gaussians[
                ..., encoded_state_dim:
            ],
            "reconstructed_state_prime_diff_mags": jnp.ravel(
                jnp.abs(reconstructed_states_prime - states[1:])
            ),
        },
        "state": {
            "state_reconstruction_diff_mags": state_reconstruction_diff_mags,
            "state_pos": jnp.ravel(jnp.abs(states[..., :7])),
            "state_vel": jnp.ravel(jnp.abs(states[..., 7:])),
            "latent_state_mags": jnp.linalg.norm(latent_states, axis=-1),
            "latent_state_gauss_var_mags": latent_state_gaussian_vars,
            "reconstructed_state_mags": jnp.abs(reconstructed_states),
        },
        "action": {
            "latent_action_mags": jnp.abs(latent_actions),
            "action_mag": jnp.mean(jnp.abs(actions), axis=-1),
            "mean_reconstructed_action_mag": jnp.mean(
                jnp.abs(reconstructed_actions), axis=-1
            ),
        },
    }
