import jax
from jax import numpy as jnp
from jax.scipy.stats.multivariate_normal import pdf as multinorm_pdf

from einops import einsum, rearrange

from rollout import collect_rollout
from nets import encoded_state_dim, encoded_action_dim


def eval_gaussian(gaussian, point):
    dim = gaussian.shape[-1] // 2
    mean = gaussian[..., :dim]
    variance = gaussian[..., dim:]

    return multinorm_pdf(point, mean, jnp.diag(variance))


def sample_gaussian(gaussian, key):
    dim = gaussian.shape[-1] // 2
    mean = gaussian[..., :dim]
    variance = gaussian[..., dim:]

    result = jax.random.multivariate_normal(key, mean, jnp.diag(variance))

    return result


def gauss_kl(p, q):
    dim = p.shape[-1] // 2
    p_var, q_var = p[..., dim:], q[..., dim:]
    p_mean, q_mean = p[..., :dim], q[..., :dim]

    ratio = p_var / q_var
    mahalanobis = jnp.square(q_mean - p_mean) / q_var

    return 0.5 * (jnp.sum(ratio) + jnp.sum(mahalanobis) - dim + jnp.sum(jnp.log(ratio)))


def loss_forward(
    state_encoder_params,
    action_encoder_params,
    transition_params,
    state_encoder_state,
    action_encoder_state,
    transition_state,
    rollout_result,
    key,
):
    states, actions = rollout_result

    states = rearrange(states, "r t d -> (r t) d")
    actions = rearrange(actions, "r t d -> (r t) d")

    encoded_state_gaussians = jax.vmap(
        state_encoder_state.apply_fn,
        (None, 0),
    )({"params": state_encoder_params}, states)

    encoded_action_gaussians = jax.vmap(
        action_encoder_state.apply_fn,
        (None, 0),
    )({"params": action_encoder_params}, actions)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[:-1])
    sampled_encoded_states = jax.vmap(sample_gaussian, (0, 0))(
        encoded_state_gaussians, rngs
    )
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[:-1])
    sampled_encoded_actions = jax.vmap(sample_gaussian, (0, 0))(
        encoded_action_gaussians, rngs
    )

    sample_encoded_state_actions = jnp.concatenate(
        [sampled_encoded_states, sampled_encoded_actions], axis=-1
    )

    transition_model_output = jax.vmap(
        transition_state.apply_fn,
        (None, 0),
    )({"params": transition_params}, sample_encoded_state_actions)

    next_state_gaussians_inference = transition_model_output[..., :-1, :]
    next_state_gaussians_ground_truth = encoded_state_gaussians[..., 1:, :]

    # next_state_gaussians_inference = rearrange(
    #     next_state_gaussians_inference, "r t d -> (r t) d"
    # )
    # next_state_gaussians_ground_truth = rearrange(
    #     next_state_gaussians_ground_truth, "r t d -> (r t) d"
    # )

    rng, key = jax.random.split(key)
    keys = jax.random.split(key, next_state_gaussians_inference.shape[:-1])
    sampled_inferred_next_states = jax.vmap(sample_gaussian, (0, 0))(
        next_state_gaussians_inference, keys
    )

    probs = jax.vmap(eval_gaussian, in_axes=(0, 0))(
        next_state_gaussians_ground_truth, sampled_inferred_next_states
    )

    return -jnp.sum(probs)


def loss_reconstruction(
    state_encoder_params,
    action_encoder_params,
    state_decoder_params,
    action_decoder_params,
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    rollout_results,
    rng,
):
    states, actions = rollout_results
    states = rearrange(states, "r t d -> (r t) d")
    actions = rearrange(actions, "r t d -> (r t) d")

    encoded_state_gaussians = state_encoder_state.apply_fn(
        {"params": state_encoder_params}, states
    )
    encoded_action_gaussians = jax.vmap(
        action_encoder_state.apply_fn,
        (None, 0),
    )({"params": action_encoder_params}, actions)

    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, states.shape[0])
    sampled_encoded_states = jax.vmap(
        sample_gaussian,
        (0, 0),
    )(encoded_state_gaussians, keys)

    rng, key = jax.random.split(rng)
    keys = jax.random.split(key, states.shape[0])
    sampled_encoded_actions = jax.vmap(
        sample_gaussian,
        (0, 0),
    )(encoded_action_gaussians, keys)
    # TODO this loss is weird
    reconstructed_state_gaussians = jax.vmap(
        action_encoder_state.apply_fn,
        (None, 0),
    )(
        state_decoder_state.apply_fn(
            {"params": state_decoder_params},
            sampled_encoded_states,
        )
    )
    reconstructed_action_gaussians = jax.vmap(
        jax.vmap(action_encoder_state.apply_fn, (None, 0)), (None, 0)
    )(
        action_decoder_state.apply_fn(
            {"params": action_decoder_params},
            sampled_encoded_actions,
        )
    )

    state_probs = eval_gaussian(reconstructed_state_gaussians, states)
    action_probs = eval_gaussian(reconstructed_action_gaussians, actions)

    return -(jnp.sum(state_probs) + jnp.sum(action_probs))


# Continuity loss idea:
#   - Sample N encoded state action pairs from the encoder distributions
#   - Penalize for KL divergence between the N distributions made from the same source state action pair


def loss_smoothness(
    state_encoder_params,
    action_encoder_params,
    state_encoder_state,
    action_encoder_state,
    transition_state,
    rollout_results,
    rng,
    samples_per_pair=16,
):
    def do_single_pair(state_gaussian, action_gaussian, rng):
        rng, key = jax.random.split(rng)
        rngs = jax.random.split(rng, samples_per_pair)

        sampled_states = jax.vmap(sample_gaussian, (None, 0))(state_gaussian, rngs)
        sampled_actions = jax.vmap(sample_gaussian, (None, 0))(action_gaussian, rngs)

        sampled_state_actions = jnp.concatenate(
            [sampled_states, sampled_actions], axis=-1
        )

        mean_state_action = jnp.concatenate(
            [
                state_gaussian[..., :encoded_state_dim],
                action_gaussian[..., :encoded_action_dim],
            ],
            axis=-1,
        )

        mean_next_state_gaussian = transition_state.apply_fn(
            {"params": transition_state.params}, mean_state_action
        )

        sampled_next_state_gaussians = jax.vmap(
            transition_state.apply_fn,
            in_axes=(None, 0, 0),
        )({"params": transition_state.params}, sampled_state_actions)

        rng, key = jax.random.split(rng)
        rngs = jax.random.split(rng, samples_per_pair)
        double_sampled_next_states = jax.vmap(
            jax.vmap(sample_gaussian, (0, 0)),
            in_axes=(0, None),
        )(sampled_next_state_gaussians, rngs)

        probs = jax.vmap(eval_gaussian, in_axes=(None, 0))(
            mean_next_state_gaussian, double_sampled_next_states
        )

        return -jnp.sum(probs)

    states, actions = rollout_results

    encoded_state_gaussians = jax.vmap(
        jax.vmap(state_encoder_state.apply_fn, (None, 0)), (None, 0)
    )({"params": state_encoder_params}, states)
    encoded_action_gaussians = jax.vmap(
        jax.vmap(action_encoder_state.apply_fn, (None, 0)), (None, 0)
    )({"params": action_encoder_params}, actions)

    encoded_state_gaussians = rearrange(encoded_state_gaussians, "r t d -> (r t) d")
    encoded_action_gaussians = rearrange(encoded_action_gaussians, "r t d -> (r t) d")

    rng, key = jax.random.split(rng)
    rngs = jax.random.split(key, encoded_state_gaussians.shape[:-1])

    losses = jax.vmap(
        jax.vmap(do_single_pair, (0, 0, 0)),
        in_axes=(0, 0, None),
    )(
        encoded_state_gaussians,
        encoded_action_gaussians,
        rngs,
    )

    return jnp.sum(losses)
