import jax
from jax import numpy as jnp

from nets import encoded_action_dim

from einops import einsum

from training import collect_latent_rollout


def score_ep_actions(
    state_z_f, state_z_0, mass_config, shape_config, action, transition_model_state, dt
):
    latent_rollout = collect_latent_rollout(
        state_z_0,
        mass_config,
        shape_config,
        action,
        transition_model_state,
        dt,
    )

    final_diff = latent_rollout[-1] - state_z_f
    return einsum(final_diff, final_diff, "i, i ->")


def make_guess(
    state_z_f,
    state_z_0,
    mass_config,
    shape_config,
    transition_model_state,
    dt,
    window,
    rng,
):
    rng, key = jax.random.split(rng)
    actions = jax.random.normal(
        key,
        (
            window,
            encoded_action_dim,
        ),
    )

    return actions


def make_best_guess(
    state_z_f,
    state_z_0,
    mass_config,
    shape_config,
    transition_model_state,
    dt,
    window,
    rng,
    guess_count=128,
):
    rng, key = jax.random.split(rng)
    guess_keys = jax.random.split(key, guess_count)

    guesses = jax.vmap(
        make_guess,
        in_axes=(None) * 7 + (0,),
    )(
        state_z_f,
        state_z_0,
        mass_config,
        shape_config,
        transition_model_state,
        dt,
        window,
        guess_keys,
    )

    guess_scores = jax.vmap(score_ep_actions, in_axes=(None) * 4 + (0,) + (None) * 2)(
        state_z_f,
        state_z_0,
        mass_config,
        shape_config,
        guesses,
        transition_model_state,
        dt,
    )
    best_guess_idx = jnp.argmin(guess_scores)

    return guesses[best_guess_idx]


def stupid_policy(
    state_encoder_state,
    state_decoder_state,
    transition_model_state,
    state,
    target_state,
    window,
    rng,
    mass_config,
    shape_config,
    dt,
    guess_count=128,
    refine_steps=20,
):
    """This policy shoots 128 random actions at the environment, refines them twice,
    and returns the one with the highest reward."""

    # Generate random actions
    rng, key = jax.random.split(rng)
    actions = jax.random.normal(key, (guess_count, window, encoded_action_dim))

    state_z_0 = state_encoder_state.apply_fn({"params": state_encoder_state}, state)
    state_z_f = state_decoder_state.apply_fn(
        {"params": state_encoder_state}, target_state
    )

    best_guess = make_best_guess(
        state_z_f,
        state_z_0,
        mass_config,
        shape_config,
        transition_model_state,
        dt,
        window,
        rng,
        guess_count=guess_count,
    )

    def scanf(action, _):
        act_grad = jax.grad(score_ep_actions, 4)(action)

        new_action = action - 0.1 * act_grad
        return new_action, _

    result, _ = jax.lax.scan(scanf, best_guess, None, length=refine_steps)
    
    return result
