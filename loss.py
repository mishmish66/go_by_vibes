import jax
from jax import numpy as jnp
from jax.scipy.stats.multivariate_normal import pdf as multinorm_pdf
from jax.scipy.stats.multivariate_normal import logpdf as multinorm_logpdf
from jax.nn import sigmoid

from einops import einsum, rearrange

from rollout import collect_rollout
from nets import encoded_state_dim, encoded_action_dim


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
    dt,
    key,
):
    states, actions = rollout_result
    
    actions = actions[:-1]

    encoded_state_gaussians = jax.vmap(
        jax.vmap(state_encoder_state.apply_fn, (None, 0)), (None, 0)
    )({"params": state_encoder_params}, states)

    encoded_action_gaussians = jax.vmap(
        jax.vmap(action_encoder_state.apply_fn, (None, 0)), (None, 0)
    )({"params": action_encoder_params}, actions)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[:-1])
    sampled_encoded_states = jax.vmap(jax.vmap(sample_gaussian, (0, 0)))(
        encoded_state_gaussians, rngs
    )
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[:-1])
    sampled_encoded_actions = jax.vmap(jax.vmap(sample_gaussian, (0, 0)))(
        encoded_action_gaussians, rngs
    )

    traj_length = states.shape[1]
    triangle_mask = jnp.tri(traj_length)

    state_times = jnp.arange(traj_length) * dt
    action_times = jnp.arange(traj_length - 1) * dt

    transition_model_output = jax.vmap(
        jax.vmap(
            transition_state.apply_fn,
            (None, None, None, None, None, 0),
        ),
        (None, 0, 0, 0, 0, None),
    )(
        {"params": transition_params},
        sampled_encoded_states,
        sampled_encoded_actions,
        state_times,
        action_times,
        triangle_mask,
    )

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

    probs = jax.vmap(eval_log_gaussian, in_axes=(0, 0))(
        next_state_gaussians_ground_truth, sampled_inferred_next_states
    )

    return -jnp.sum(sigmoid(probs)) / sum(probs.shape[:-1])


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

    encoded_state_gaussians = jax.vmap(
        state_encoder_state.apply_fn,
        (None, 0),
    )({"params": state_encoder_params}, states)

    rng, key = jax.random.split(rng)
    rngs = jax.random.split(rng, (2, states.shape[0]))
    
    sampled_encoded_states = jax.vmap(
        sample_gaussian,
        (0, 0),
    )(encoded_state_gaussians, rngs[0])
    
    encoded_action_gaussians = jax.vmap(
        action_encoder_state.apply_fn,
        (None, 0, 0),
    )({"params": action_encoder_params},
      actions,
      sampled_encoded_states,
    )
    
    sampled_encoded_actions = jax.vmap(
        sample_gaussian,
        (0, 0),
    )(encoded_action_gaussians, rngs[1])

    reconstructed_state_gaussians = jax.vmap(
        state_decoder_state.apply_fn,
        (None, 0),
    )(
        {"params": state_decoder_params},
        sampled_encoded_states,
    )

    reconstructed_action_gaussians = jax.vmap(
        action_decoder_state.apply_fn,
        (None, 0),
    )(
        {"params": action_decoder_params},
        sampled_encoded_actions,
        sampled_encoded_states,
    )

    state_probs = jax.vmap(
        eval_log_gaussian,
        (0, 0),
    )(reconstructed_state_gaussians, states)
    action_probs = jax.vmap(
        eval_log_gaussian,
        (0, 0),
    )(reconstructed_action_gaussians, actions)

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
        rngs = jax.random.split(rng, (2, samples_per_pair))

        # Sample state deltas from a unit normal distribution
        sampled_states = jax.vmap(jax.random.multivariate_normal, (0, None, None))(
            rngs[0], state_gaussian[..., encoded_state_dim:], jnp.eye(encoded_state_dim)
        )
        sampled_actions = jax.vmap(jax.random.multivariate_normal, (0, None, None))(
            rngs[1],
            action_gaussian[..., encoded_action_dim:],
            jnp.eye(encoded_action_dim),
        )

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
        rng, key = jax.random.split(key)
        sampled_center_next_state = sample_gaussian(mean_next_state_gaussian, rng)

        sampled_next_state_gaussians = jax.vmap(
            transition_state.apply_fn,
            (None, 0),
        )({"params": transition_state.params}, sampled_state_actions)

        rng, key = jax.random.split(rng)
        rngs = jax.random.split(rng, samples_per_pair)
        double_sampled_next_states = jax.vmap(
            sample_gaussian,
            (0, 0),
        )(sampled_next_state_gaussians, rngs)

        # Evaluate the probability of the other state under a unit variance distributions
        probs = jax.vmap(
            multinorm_pdf,
            (0, None, None),
        )(
            double_sampled_next_states,
            sampled_center_next_state,
            jnp.eye(encoded_state_dim),
        )

        return -jnp.sum(probs)

    states, actions = rollout_results

    states = rearrange(states, "r t d -> (r t) d")
    actions = rearrange(actions, "r t d -> (r t) d")

    encoded_state_gaussians = jax.vmap(state_encoder_state.apply_fn, (None, 0))(
        {"params": state_encoder_params}, states
    )
    encoded_action_gaussians = jax.vmap(action_encoder_state.apply_fn, (None, 0))(
        {"params": action_encoder_params}, actions
    )

    rng, key = jax.random.split(rng)
    rngs = jax.random.split(key, encoded_state_gaussians.shape[:-1])

    losses = jax.vmap(do_single_pair, (0, 0, 0))(
        encoded_state_gaussians,
        encoded_action_gaussians,
        rngs,
    )

    return jnp.sum(losses)


def loss_state_regularization(
    state_encoder_params,
    state_encoder_state,
    action_encoder_state,
    rollout_results,
    dt,
    rng,
):
    # Encode states and actions
    states, actions = rollout_results
    encoded_state_gaussians = jax.vmap(state_encoder_state.apply_fn, (None, 0))(
        {"params": state_encoder_params}, states
    )
    encoded_action_gaussians = jax.vmap(action_encoder_state.apply_fn, (None, 0))(
        {"params": action_encoder_state.params}, actions
    )

    # Get next and previous states and actions by throwing out first and last
    next_state_gaussians = encoded_state_gaussians[..., 1:, :]
    prev_state_gaussians = encoded_state_gaussians[..., :-1, :]
    next_action_gaussians = encoded_action_gaussians[..., 1:, :]
    prev_action_gaussians = encoded_action_gaussians[..., :-1, :]

    next_state_gaussians = rearrange(next_state_gaussians, "r t d -> (r t) d")
    prev_state_gaussians = rearrange(prev_state_gaussians, "r t d -> (r t) d")
    next_action_gaussians = rearrange(next_action_gaussians, "r t d -> (r t) d")
    prev_action_gaussians = rearrange(prev_action_gaussians, "r t d -> (r t) d")

    # Sample a point from each
    rng, key = jax.random.split(rng)
    rngs = jax.random.split(
        rng,
        (2, *next_state_gaussians.shape[:-1]),
    )
    encoded_next_states = jax.vmap(sample_gaussian, (0, 0))(
        next_state_gaussians, rngs[0]
    )
    encoded_prev_states = jax.vmap(sample_gaussian, (0, 0))(
        prev_state_gaussians, rngs[1]
    )

    # Enforce similarity of states and next states
    state_loss = -jax jnp.abs(
        jnp.linalg.norm(encoded_prev_states - encoded_next_states) - dt,
    )

    return jnp.sum(state_loss)


def loss_action_regularization(
    action_encoder_params,
    state_encoder_state,
    action_encoder_state,
    transition_state,
    rollout_results,
    action_neighborhood_size,
    rng,
):
    # Encode states and actions
    states, actions = rollout_results
    encoded_state_gaussians = jax.vmap(state_encoder_state.apply_fn, (None, 0))(
        {"params": state_encoder_state.params}, states
    )
    encoded_action_gaussians = jax.vmap(action_encoder_state.apply_fn, (None, 0))(
        {"params": action_encoder_params}, actions
    )

    # Get next and previous states and actions by throwing out first and last
    next_state_gaussians = encoded_state_gaussians[..., 1:, :]
    prev_state_gaussians = encoded_state_gaussians[..., :-1, :]
    next_action_gaussians = encoded_action_gaussians[..., 1:, :]
    prev_action_gaussians = encoded_action_gaussians[..., :-1, :]

    next_state_gaussians = rearrange(next_state_gaussians, "r t d -> (r t) d")
    prev_state_gaussians = rearrange(prev_state_gaussians, "r t d -> (r t) d")
    next_action_gaussians = rearrange(next_action_gaussians, "r t d -> (r t) d")
    prev_action_gaussians = rearrange(prev_action_gaussians, "r t d -> (r t) d")

    # Sample a point from each
    rng, key = jax.random.split(rng)
    rngs = jax.random.split(
        rng,
        (5, *next_state_gaussians.shape[:-1]),
    )
    encoded_next_states = jax.vmap(sample_gaussian, (0, 0))(
        next_state_gaussians, rngs[0]
    )
    encoded_prev_states = jax.vmap(sample_gaussian, (0, 0))(
        prev_state_gaussians, rngs[1]
    )
    encoded_prev_actions = jax.vmap(sample_gaussian, (0, 0))(
        prev_action_gaussians, rngs[2]
    )

    # Sample actions in neighborhood of prev actions
    similar_prev_actions = jax.vmap(jax.random.multivariate_normal, (0, 0, None))(
        rngs[3], encoded_prev_actions, jnp.eye(encoded_action_dim)
    )
    similar_state_actions = jnp.concatenate(
        [encoded_prev_states, similar_prev_actions], axis=-1
    )
    # Step the prev states through the similar actions
    post_similar_state_action_gaussians = jax.vmap(
        transition_state.apply_fn,
        (None, 0),
    )(
        {"params": transition_state.params},
        similar_state_actions,
    )
    sampled_post_similar_state_actions = jax.vmap(
        sample_gaussian,
        (0, 0),
    )(post_similar_state_action_gaussians, rngs[4])

    # Penalize for leaving the action neighborhood
    dists = jnp.linalg.norm(sampled_post_similar_state_actions - encoded_next_states)
    neighborhood_violation = dists - action_neighborhood_size
    action_loss = -jnp.square(jnp.clip(neighborhood_violation, a_min=0, a_max=None))

    return jnp.sum(action_loss)
