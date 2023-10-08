import jax
from jax import numpy as jnp
from jax.scipy.stats.multivariate_normal import pdf as multinorm_pdf
from jax.scipy.stats.multivariate_normal import logpdf as multinorm_logpdf
from jax.nn import sigmoid

from einops import einsum, rearrange

from .rollout import collect_rollout
from .nets import (
    encoded_state_dim,
    encoded_action_dim,
)

from .vibe_state import VibeState, TrainConfig

from .inference import (
    sample_gaussian,
    get_latent_state_gaussian,
    get_latent_action_gaussian,
    get_latent_state_prime_gaussians,
    get_state_space_gaussian,
    get_action_space_gaussian,
    make_mask,
)


def eval_gaussian(gaussian, point):
    dim = gaussian.shape[-1] // 2
    mean = gaussian[..., :dim]
    variance = gaussian[..., dim:]

    return multinorm_pdf(point, mean, jnp.diag(variance))


def eval_log_gaussian(gaussian, point):
    dim = gaussian.shape[-1] // 2
    mean = gaussian[..., :dim]
    variance = gaussian[..., dim:]

    return multinorm_logpdf(point, mean, jnp.diag(variance))


def loss_forward(
    inferred_latent_states_prime,
    gt_next_latent_states,
):
    diffs = inferred_latent_states_prime - gt_next_latent_states

    diff_l2s = jnp.linalg.norm(diffs, axis=-1)

    # return jnp.nan_to_num(jnp.mean(scaled_diff_log_mags), nan=1e9)
    return jnp.mean(diff_l2s)


def loss_reconstruction(
    inferred_state_gaussian_params,
    inferred_action_gaussian_params,
    gt_states,
    gt_actions,
):
    state_probs = jax.vmap(
        eval_log_gaussian,
        (0, 0),
    )(inferred_state_gaussian_params, gt_states)
    action_probs = jax.vmap(
        eval_log_gaussian,
        (0, 0),
    )(inferred_action_gaussian_params, gt_actions)

    # scaled_state_probs = jnp.nan_to_num(state_probs, nan=-1e20)
    # scaled_action_probs = jnp.nan_to_num(action_probs, nan=-1e20)

    return -(jnp.mean(state_probs) + jnp.mean(action_probs))


def loss_smoothness(
    original_latent_states,
    neighborhood_result_latent_states,
):
    diffs = original_latent_states - neighborhood_result_latent_states
    diff_mags = jnp.linalg.norm(diffs, axis=-1)
    diff_mags_dist_from_1 = jnp.abs(diff_mags - 1)

    return jnp.mean(diff_mags_dist_from_1)


def loss_disperse(
    latent_states,
):
    diffs = latent_states[None, ...] - latent_states[:, None, ...]
    diff_mags = jnp.linalg.norm(diffs, axis=-1)

    return -jnp.mean(diff_mags)


def loss_condense(
    latent_actions,
):
    diffs = latent_actions[None, ...] - latent_actions[:, None, ...]
    diff_mags = jnp.linalg.norm(diffs, axis=-1)

    return jnp.mean(diff_mags)


def composed_loss(
    key,
    states,
    actions,
    vibe_state: VibeState,
    train_config: TrainConfig,
    reconstruction_loss_num_encoder_samples=64,
    forward_loss_num_encoder_samples=1,
    forward_loss_num_random_indices=16,
    forward_loss_num_forward_samples=4,
):
    jax.debug.print("states shape: {}", states.shape)
    latent_state_gaussian_params = jax.vmap(
        get_latent_state_gaussian,
        (0, None, None),
    )(
        states,
        vibe_state,
        train_config,
    )

    reconstruction_loss_buffer = []
    for _ in range(reconstruction_loss_num_encoder_samples):
        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, states.shape[0])
        latent_states = jax.vmap(sample_gaussian, (0, 0))(
            rngs, latent_state_gaussian_params
        )
        latent_action_gaussian_params = jax.vmap(
            get_latent_action_gaussian,
            (0, 0, None, None),
        )(
            actions,
            latent_states[:-1],
            vibe_state,
            train_config,
        )
        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, actions.shape[0])
        latent_actions = jax.vmap(sample_gaussian, (0, 0))(
            rngs, latent_action_gaussian_params
        )

        state_space_gaussians = jax.vmap(
            get_state_space_gaussian,
            in_axes=(0, None, None),
        )(latent_states, vibe_state, train_config)
        action_space_gaussians = jax.vmap(
            get_action_space_gaussian,
            in_axes=(0, 0, None, None),
        )(latent_actions, latent_states[:-1], vibe_state, train_config)

        # Evaluate reconstruction loss:
        reconstruction_loss = loss_reconstruction(
            state_space_gaussians,
            action_space_gaussians,
            states,
            actions,
        )
        reconstruction_loss_buffer.append(reconstruction_loss)

    reconstruction_loss = sum(reconstruction_loss_buffer) / len(
        reconstruction_loss_buffer
    )

    # Evaluate forward loss:
    forward_loss_buffer = []
    for _ in range(forward_loss_num_encoder_samples):
        # Sample latent states and actions
        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, states.shape[0])
        latent_states = jax.vmap(sample_gaussian, (0, 0))(
            rngs, latent_state_gaussian_params
        )
        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, actions.shape[0])
        latent_actions = jax.vmap(sample_gaussian, (0, 0))(
            rngs, latent_action_gaussian_params
        )

        # Infer next latent states
        for _ in range(forward_loss_num_random_indices):
            prev_latent_states = latent_states[:-1]
            next_latent_states = latent_states[1:]

            # Generate a random index to make the network guess
            # the second part of the latent state sequence
            # give it some padding so it has to predict at least
            # an eighth of the sequence
            end_padding = prev_latent_states.shape[0] // 8
            random_i = jax.random.randint(
                key, [], 0, prev_latent_states.shape[0] - end_padding
            )

            # Now just a bunch of stuff to slice up the arrays
            known_state_count = random_i

            known_prev_latent_state_mask = make_mask(
                prev_latent_states.shape[0], known_state_count
            )
            inferred_next_state_mask = 1 - known_prev_latent_state_mask

            # Now we get predict the next latent states
            latent_states_prime_gaussians = get_latent_state_prime_gaussians(
                prev_latent_states,
                latent_actions,
                train_config.dt,
                vibe_state,
                train_config,
                known_prev_latent_state_mask,
            )

            gt_next_latent_states = einsum(
                next_latent_states,
                1 - known_prev_latent_state_mask,
                "t d, t -> t d",
            )

            for _ in range(forward_loss_num_forward_samples):
                # Sample the next latent states
                rng, key = jax.random.split(key)
                rngs = jax.random.split(rng, latent_states_prime_gaussians.shape[0])
                latent_states_prime_sampled = jax.vmap(
                    sample_gaussian, (0, 0)
                )(
                    rngs,
                    latent_states_prime_gaussians,
                )
                inferred_latent_states_prime_sampled = einsum(
                    latent_states_prime_sampled,
                    inferred_next_state_mask,
                    "t d, t -> t d",
                )

                # Evaluate the forward loss
                forward_loss = loss_forward(
                    inferred_latent_states_prime_sampled, gt_next_latent_states
                )

                forward_loss_buffer.append(forward_loss)

    forward_loss = sum(forward_loss_buffer) / len(forward_loss_buffer)

    return (
        train_config.reconstruction_weight * reconstruction_loss
        + train_config.forward_weight * forward_loss
    ), {
        "reconstruction_loss": reconstruction_loss,
        "forward_loss": forward_loss,
    }
