import jax
from jax import numpy as jnp

from training.nets import (
    encoded_state_dim,
    encoded_action_dim,
)

from einops import einsum

from training.inference import infer_states, make_mask

# from training.vibe_state import collect_latent_rollout

from training.loss import sample_gaussian


def random_action(key, action_bounds):
    rng, key = jax.random.split(key)
    random_nums = jax.random.uniform(rng, (action_bounds.shape[0],))
    scaled = random_nums * (action_bounds[:, 1] - action_bounds[:, 0])
    scaled_and_shifted = scaled + action_bounds[:, 0]

    return scaled_and_shifted


def random_policy(
    key,
    start_state,
    action_i,
    vibe_state,
    vibe_config,
):
    """This policy just samples random actions from the action space."""

    rng, key = jax.random.split(key)
    return random_action(rng, vibe_config.env_config.action_bounds)


# Down here is not really ready, but I'm just putting it here for now
def make_traj_opt_policy(cost_func):
    def policy(key, start_state, vibe_state, vibe_config):
        pass


def max_dist_policy(
    key,
    start_state,
    vibe_state,
    vibe_config,
    window=64,
    guess_count=128,
    refine_steps=256,
):
    """This policy shoots 128 random actions at the environment, refines them,
    and returns the one with the highest reward."""

    def max_dist_objective(
        latent_actions,
        key,
    ):
        latent_states = jnp.zeros(
            (latent_actions.shape[0], vibe_config.env_config.state_dim)
        )
        latent_rollout = infer_states(
            key,
            latent_states,
            latent_actions,
            vibe_state,
            vibe_config,
            make_mask(latent_actions.shape[0], 1),
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
