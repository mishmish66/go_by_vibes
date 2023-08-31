import jax
from jax import numpy as jnp

from nets import encoded_state_dim, encoded_action_dim

from einops import einsum

from training import collect_latent_rollout

from loss import sample_gaussian


def score_ep_actions(
    state_z_f,
    state_z_0,
    mass_config,
    shape_config,
    action,
    transition_model_state,
    dt,
    key,
):
    latent_rollout = collect_latent_rollout(
        state_z_0, mass_config, shape_config, action, transition_model_state, dt, key
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
    key,
    guess_count=128,
):
    rng, key = jax.random.split(key)
    guess_rngs = jax.random.split(rng, guess_count)

    guesses = jax.vmap(
        make_guess,
        in_axes=(None,) * 7 + (0,),
    )(
        state_z_f,
        state_z_0,
        mass_config,
        shape_config,
        transition_model_state,
        dt,
        window,
        guess_rngs,
    )

    rng, key = jax.random.split(key)
    guess_rngs = jax.random.split(rng, guess_count)

    guess_scores = jax.vmap(
        score_ep_actions, in_axes=(None,) * 4 + (0,) + (None,) * 2 + (0,)
    )(
        state_z_f,
        state_z_0,
        mass_config,
        shape_config,
        guesses,
        transition_model_state,
        dt,
        guess_rngs,
    )
    best_guess_idx = jnp.argmin(guess_scores)

    return guesses[best_guess_idx]


def stupid_policy(
    state_encoder_state,
    action_decoder_state,
    transition_model_state,
    state,
    target_state,
    window,
    key,
    mass_config,
    shape_config,
    dt,
    guess_count=128,
    refine_steps=20,
):
    """This policy shoots 128 random actions at the environment, refines them,
    and returns the one with the highest reward."""

    state_z_0_gaussian = state_encoder_state.apply_fn(
        {"params": state_encoder_state.params}, state
    )
    state_z_s_gaussian = state_encoder_state.apply_fn(
        {"params": state_encoder_state.params}, target_state
    )

    state_z_0_mean = state_z_0_gaussian[..., :encoded_state_dim]
    state_z_s_mean = state_z_s_gaussian[..., :encoded_state_dim]

    rng, key = jax.random.split(key)

    best_guess = make_best_guess(
        state_z_s_mean,
        state_z_0_mean,
        mass_config,
        shape_config,
        transition_model_state,
        dt,
        window,
        key,
        guess_count=guess_count,
    )

    def scanf(carry, _):
        action, key = carry
        rng, key = jax.random.split(key)

        act_grad = jax.grad(score_ep_actions, 4)(
            state_z_s_mean,
            state_z_0_mean,
            mass_config,
            shape_config,
            action,
            transition_model_state,
            dt,
            rng,
        )

        new_action = action - 0.1 * act_grad
        return (new_action, key), _
    

    rng, key = jax.random.split(key)
    (encoded_action_sequence, _), _ = jax.lax.scan(
        scanf, (best_guess, rng), None, length=refine_steps
    )

    encoded_next_action = encoded_action_sequence[0]

    rng, key = jax.random.split(key)
    next_action_gaussian = action_decoder_state.apply_fn(
        {"params": action_decoder_state.params}, encoded_next_action
    )

    sampled_next_action = sample_gaussian(next_action_gaussian, rng)

    return sampled_next_action
