import jax
from jax import numpy as jnp

from nets import (
    encoded_state_dim,
    encoded_action_dim,
    encode_state,
    encode_action,
    get_action_space_gaussian,
)

from einops import einsum

from training import collect_latent_rollout

from loss import sample_gaussian

def random_policy(
    key,
    action_bounds,
):
    """This policy just samples random actions from the action space."""

    rng, key = jax.random.split(key)
    return (jax.random.uniform(rng, (action_bounds.shape[0],)) * 2 - 1) * action_bounds

def max_dist_policy(
    key,
    state_encoder_state,
    action_decoder_state,
    transition_model_state,
    start_state,
    window,
    dt,
    guess_count=128,
    refine_steps=20,
):
    """This policy shoots 128 random actions at the environment, refines them,
    and returns the one with the highest reward."""

    def max_dist_objective(
        key,
        latent_actions,
    ):
        latent_rollout = collect_latent_rollout(
            key,
            transition_model_state,
            latent_start_state,
            latent_actions,
            dt,
        )

        diffs = latent_rollout - latent_start_state
        diff_mag = jnp.linalg.norm(diffs, axis=-1)

        return jnp.sum(diff_mag)

    rng, key = jax.random.split(key)
    latent_start_state = encode_state(rng, state_encoder_state, start_state)

    rng, key = jax.random.split(key)
    latent_action_guesses = jax.random.normal(
        rng,
        (
            guess_count,
            window,
            encoded_action_dim,
        ),
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, latent_action_guesses.shape[0])
    guess_scores = jax.vmap(max_dist_objective, in_axes=(0, 0))(
        rngs, latent_action_guesses
    )

    best_guess_i = jnp.argmax(guess_scores)

    best_guess = latent_action_guesses[best_guess_i]

    def scanf(current_plan, key):
        rng, key = jax.random.split(key)

        act_grad = jax.grad(max_dist_objective, 1)(
            rng,
            current_plan,
        )

        next_plan = current_plan - 0.1 * act_grad
        return next_plan, None

    rng, key = jax.random.split(key)
    encoded_action_sequence, _ = jax.lax.scan(
        scanf, best_guess, jax.random.split(rng, refine_steps)
    )

    latent_next_action = encoded_action_sequence[0]

    next_action_gaussian = get_action_space_gaussian(
        action_decoder_state, latent_next_action, latent_start_state
    )

    rng, key = jax.random.split(key)
    sampled_next_action = sample_gaussian(rng, next_action_gaussian)

    return sampled_next_action
