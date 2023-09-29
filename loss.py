import jax
from jax import numpy as jnp
from jax.scipy.stats.multivariate_normal import pdf as multinorm_pdf
from jax.scipy.stats.multivariate_normal import logpdf as multinorm_logpdf
from jax.nn import sigmoid

from einops import einsum, rearrange

from rollout import collect_rollout
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
    key,
    state_encoder_state,
    action_encoder_state,
    transition_state,
    rollout_result,
    dt,
    state_encoder_params=None,
    action_encoder_params=None,
    transition_params=None,
):
    if state_encoder_params is None:
        state_encoder_params = state_encoder_state.params
    if action_encoder_params is None:
        action_encoder_params = action_encoder_state.params
    if transition_params is None:
        transition_params = transition_state.params

    states, actions = rollout_result

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[:-1])
    latent_states = jax.vmap(
        jax.vmap(encode_state, (0, None, 0, None)), (0, None, 0, None)
    )(rngs, state_encoder_state, states, state_encoder_params)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, actions.shape[:-1])
    latent_actions = jax.vmap(
        jax.vmap(encode_action, (0, None, 0, 0, None)), (0, None, 0, 0, None)
    )(rngs, action_encoder_state, actions, latent_states[:, :-1], action_encoder_params)

    def single_traj(key, latent_states, latent_actions):
        def single_i(key, known_state_mask):
            known_states = einsum(
                latent_states[:-1], known_state_mask[:-1], "... d, ... -> ... d"
            )
            rng, key = jax.random.split(key)
            # jax.debug.print("Infer states!")
            latent_states_prime = infer_states(
                rng,
                transition_state,
                known_states,
                latent_actions,
                dt,
                transition_params,
            )

            inferred_states = einsum(
                latent_states_prime, (1 - known_state_mask)[1:], "... d, ... -> ... d"
            )
            gt_states = einsum(
                latent_states[1:], (1 - known_state_mask)[1:], "... d, ... -> ... d"
            )

            indices = (
                jnp.cumsum((1 - known_state_mask)[1:].astype(jnp.float32), axis=0) - 0.5
            )
            decreasing_function = 1 / ((indices - 0.5) * (1 - known_state_mask)[1:] + 1)

            diffs = gt_states - inferred_states
            diff_mag_sq = einsum(diffs, diffs, "... d, ... d -> ...")

            bounded_diff_mag_sq = sigmoid(diff_mag_sq * decreasing_function) * 2 - 1

            return jnp.sum(bounded_diff_mag_sq) / inferred_states.shape[0]

        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, latent_states.shape[0] - 1)
        # Too much memory, need to do this in a loop
        traj_i_results = jax.vmap(single_i, (0, 0))(
            rngs,
            jnp.tri(latent_states.shape[0])[:-1],
        )
        # tri = jnp.tri(latent_states.shape[0])[:-1]
        # transposed_args = [(rngs[i], tri[i]) for i in range(rngs.shape[0])]
        # traj_i_results = jax.lax.map(
        #     jax.tree_util.Partial(single_i),
        #     transposed_args,
        # )

        return jnp.sum(traj_i_results)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[0])

    # Accumulate the loss N states at a time
    # each_traj_loss_batches = []
    # N = 4
    # i = 0
    # while i < states.shape[0]:
    #     end_i = min(i + N, states.shape[0])
    #     each_traj_loss_batches.append(
    #         jax.vmap(
    #             single_traj,
    #             (0, 0, 0),
    #         )(rngs[i : end_i], latent_states[i : end_i], latent_actions[i : end_i])
    #     )
    #     i += N
    # each_traj_loss = jnp.concatenate(each_traj_loss_batches)
    each_traj_loss = jax.vmap(single_traj, (0, 0, 0))(
        rngs, latent_states, latent_actions
    )

    return jnp.sum(each_traj_loss)


def loss_reconstruction(
    key,
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    rollout_results,
    state_encoder_params=None,
    action_encoder_params=None,
    state_decoder_params=None,
    action_decoder_params=None,
):
    if state_encoder_params is None:
        state_encoder_params = state_encoder_state.params
    if action_encoder_params is None:
        action_encoder_params = action_encoder_state.params
    if state_decoder_params is None:
        state_decoder_params = state_decoder_state.params
    if action_decoder_params is None:
        action_decoder_params = action_decoder_state.params

    states, actions = rollout_results

    states = states[:, :-1]
    states = rearrange(states, "r t d -> (r t) d")
    actions = rearrange(actions, "r t d -> (r t) d")

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[:-1])
    latent_states = jax.vmap(encode_state, (0, None, 0, None))(
        rngs, state_encoder_state, states, state_encoder_params
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, actions.shape[:-1])
    latent_actions = jax.vmap(encode_action, (0, None, 0, 0, None))(
        rngs,
        action_encoder_state,
        actions,
        latent_states,
        action_encoder_params,
    )

    state_space_gaussians = jax.vmap(get_state_space_gaussian, (None, 0, None))(
        state_decoder_state, latent_states, state_decoder_params
    )
    action_space_gaussians = jax.vmap(get_action_space_gaussian, (None, 0, 0, None))(
        action_decoder_state, latent_actions, latent_states, action_decoder_params
    )

    state_probs = jax.vmap(
        eval_log_gaussian,
        (0, 0),
    )(state_space_gaussians, states)
    action_probs = jax.vmap(
        eval_log_gaussian,
        (0, 0),
    )(action_space_gaussians, actions)

    bounded_scaled_state_probs = sigmoid(state_probs) * 2 - 1
    bounded_scaled_action_probs = sigmoid(action_probs) * 2 - 1

    return -(jnp.sum(bounded_scaled_state_probs) + jnp.sum(bounded_scaled_action_probs))


# Continuity loss idea:
#   - Sample N encoded state action pairs from the encoder distributions
#   - Penalize for KL divergence between the N distributions made from the same source state action pair


def loss_smoothness(
    key,
    state_encoder_state,
    action_encoder_state,
    transition_state,
    rollout_results,
    dt,
    state_encoder_params=None,
    action_encoder_params=None,
):
    if state_encoder_params is None:
        state_encoder_params = state_encoder_state.params
    if action_encoder_params is None:
        action_encoder_params = action_encoder_state.params

    def loss_smoothness_one_trajectory(
        key,
        latent_states,
        latent_actions,
    ):
        # Define the loss for one set of neighborhood actions

        def one_latent_trajectory_actions(key, known_state_mask):
            same_actions_mask = known_state_mask[1:]
            same_actions = einsum(
                latent_actions, same_actions_mask, "... d, ... -> ... d"
            )

            rng, key = jax.random.split(key)
            rngs = jax.random.split(rng, latent_actions.shape[0])
            neighborhood_actions = jax.vmap(
                jax.random.multivariate_normal,
                (0, 0, None),
            )(
                rngs,
                latent_actions,
                jnp.eye(latent_actions.shape[-1]),
            )
            new_neighborhood_actions = einsum(
                neighborhood_actions, (1 - same_actions_mask), "... d, ... -> ... d"
            )

            new_action_sequence = same_actions + new_neighborhood_actions

            latent_source_states = latent_states[:-1]
            known_states = einsum(
                latent_source_states, known_state_mask[:-1], "... d, ... -> ... d"
            )

            # Run the neighborhood actions through the transition model
            rng, key = jax.random.split(key)
            latent_states_prime = infer_states(
                rng,
                transition_state,
                known_states,
                new_action_sequence,
                dt,
            )

            # Get the ground truth latent states related to the inferred latent states
            unknown_next_state_mask = 1 - known_state_mask[1:]
            gt_latent_next_states = latent_states[1:]
            gt_latent_states = einsum(
                gt_latent_next_states, unknown_next_state_mask, "... d, ... -> ... d"
            )
            # Grab the inferred latent states and not the ones that are masked
            inferred_latent_states = einsum(
                latent_states_prime, unknown_next_state_mask, "... d, ... -> ... d"
            )
            # Find the difference between gt and inferred latent states
            diffs = gt_latent_states - inferred_latent_states
            # Find the squared magnitude
            diff_mag_sq = einsum(diffs, diffs, "... d, ... d -> ...")
            # Find the deviation from the neighborhood
            neighborhood_violation = jnp.clip(diff_mag_sq - 1, a_min=0)
            indices = jnp.cumsum(unknown_next_state_mask.astype(jnp.float32))
            decreasing_function = 1 / (indices - 0.5) * (unknown_next_state_mask)
            bounded_scaled_diffs = (
                sigmoid(neighborhood_violation) * 2 - 1
            ) * decreasing_function
            bounded_result = sigmoid(jnp.sum(bounded_scaled_diffs))
            return bounded_result

        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, latent_states.shape[0])

        action_smoothness_loss_per_mask = jax.vmap(
            one_latent_trajectory_actions,
            (0, 0),
        )(rngs, jnp.tri(latent_states.shape[0]))
        action_smoothness_loss = jnp.sum(action_smoothness_loss_per_mask)

        # Now we will do the same thing but with the latent states

        def one_latent_trajectory_states(key, known_state_mask):
            # Randomize the last state in the neighborhood
            same_state_mask = known_state_mask[:-1]
            same_states = einsum(
                latent_states[:-1], same_state_mask, "... d, ... -> ... d"
            )

            new_state_mask = known_state_mask[:-1] - known_state_mask[1:]
            state_to_be_randomized = einsum(
                latent_states[:-1], new_state_mask, "... d, ... -> ... d"
            )
            state_to_be_randomized = jnp.sum(state_to_be_randomized, axis=0)

            rng, key = jax.random.split(key)
            normal = jax.random.normal(
                rng,
                state_to_be_randomized.shape,
            )
            new_state = state_to_be_randomized + normal

            new_state_sequence = same_states + einsum(
                new_state, new_state_mask, "d, ... -> ... d"
            )

            # Run the neighborhood actions through the transition model
            rng, key = jax.random.split(key)
            latent_states_prime = infer_states(
                rng,
                transition_state,
                new_state_sequence,
                latent_actions,
                dt,
            )

            # Get the ground truth latent states related to the inferred latent states
            unknown_next_state_mask = 1 - known_state_mask[1:]
            gt_latent_states = einsum(
                latent_states[1:], unknown_next_state_mask, "... d, ... -> ... d"
            )
            # Grab the inferred latent states and not the ones that are masked
            inferred_latent_states = einsum(
                latent_states_prime, unknown_next_state_mask, "... d, ... -> ... d"
            )
            # Find the difference between gt and inferred latent states
            diffs = gt_latent_states - inferred_latent_states
            # Find the squared magnitude
            diff_mag_sq = einsum(diffs, diffs, "... d, ... d -> ...")
            # Find the deviation from the neighborhood
            neighborhood_violation = jnp.clip(diff_mag_sq - 1, a_min=0)
            indices = jnp.cumsum(unknown_next_state_mask.astype(jnp.float32))
            decreasing_function = 1 / (indices - 0.5) * (unknown_next_state_mask)
            bounded_scaled_diffs = sigmoid(neighborhood_violation) * decreasing_function
            bounded_result = sigmoid(jnp.sum(bounded_scaled_diffs))
            return bounded_result

        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, latent_states.shape[0])
        state_smoothness_loss_per_mask = jax.vmap(
            one_latent_trajectory_states,
            (0, 0),
        )(rngs, jnp.tri(latent_states.shape[0]))
        state_smoothness_loss = jnp.sum(state_smoothness_loss_per_mask)

        return action_smoothness_loss + state_smoothness_loss

    states, actions = rollout_results

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[:-1])
    latent_states = jax.vmap(
        jax.vmap(encode_state, (0, None, 0, None)), (0, None, 0, None)
    )(rngs, state_encoder_state, states, state_encoder_params)
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, actions.shape[:-1])
    latent_actions = jax.vmap(
        jax.vmap(encode_action, (0, None, 0, 0, None)), (0, None, 0, 0, None)
    )(rngs, action_encoder_state, actions, latent_states[:, :-1], action_encoder_params)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, latent_states.shape[0])
    losses = jax.vmap(
        loss_smoothness_one_trajectory,
        (0, 0, 0),
    )(rngs, latent_states, latent_actions)

    return jnp.sum(losses)


def loss_disperse(
    key,
    state_encoder_state,
    action_encoder_state,
    transition_state,
    rollout_results,
    action_bounds,
    dt,
    state_encoder_params=None,
    action_encoder_params=None,
):
    if state_encoder_params is None:
        state_encoder_params = state_encoder_state.params
    if action_encoder_params is None:
        action_encoder_params = action_encoder_state.params

    def loss_disperse_one_trajectory(
        key,
        latent_states,
        latent_actions,
    ):
        def one_latent_trajectory(key, known_state_mask):
            # Sample random actions at and right before unknown state
            random_action_mask = 1 - known_state_mask[1:]
            same_actions = einsum(
                latent_actions, 1 - random_action_mask, "... d, ... -> ... d"
            )

            rng, key = jax.random.split(key)
            uniform_random = (
                jax.random.uniform(
                    rng,
                    (*latent_actions.shape[:-1], action_bounds.shape[0]),
                )
                * 2
                - 1
            )
            all_actions_random = einsum(
                uniform_random, action_bounds, "... d, d -> ... d"
            )

            rng, key = jax.random.split(key)
            all_latent_random_actions = jax.vmap(encode_action, (0, None, 0, 0, None))(
                jax.random.split(rng, all_actions_random.shape[0]),
                action_encoder_state,
                all_actions_random,
                latent_states[:-1],
                action_encoder_params,
            )

            latent_random_actions = einsum(
                all_latent_random_actions, random_action_mask, "... d, ... -> ... d"
            )

            new_action_sequence = same_actions + latent_random_actions

            rng, key = jax.random.split(key)
            latent_states_prime = infer_states(
                rng, transition_state, latent_states[:-1], new_action_sequence, dt
            )

            # Find the difference between gt and inferred latent states
            unknown_next_state_mask = 1 - known_state_mask[1:]
            gt_latent_states = einsum(
                latent_states[1:], unknown_next_state_mask, "... d, ... -> ... d"
            )
            inferred_latent_states = einsum(
                latent_states_prime, unknown_next_state_mask, "... d, ... -> ... d"
            )
            diffs = gt_latent_states - inferred_latent_states
            # Find the squared magnitude
            diff_mag_sq = einsum(diffs, diffs, "... d, ... d -> ...")

            # Find the deviation from the neighborhood
            indices = jnp.cumsum(unknown_next_state_mask.astype(jnp.float32))
            decreasing_function = 1 / (indices - 0.5) * (unknown_next_state_mask)
            bounded_scaled_diffs = sigmoid(diff_mag_sq) * decreasing_function
            bounded_result = sigmoid(jnp.sum(bounded_scaled_diffs))
            return -bounded_result

        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, latent_states.shape[0])
        state_dispersion_loss_per_i = jax.vmap(one_latent_trajectory, (0, 0))(
            rngs, jnp.tri(latent_states.shape[0])
        )
        state_dispersion_loss = jnp.sum(state_dispersion_loss_per_i)

        return state_dispersion_loss

    states, actions = rollout_results

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[:-1])
    latent_states = jax.vmap(
        jax.vmap(encode_state, (0, None, 0, None)), (0, None, 0, None)
    )(rngs, state_encoder_state, states, state_encoder_params)
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, actions.shape[:-1])
    latent_actions = jax.vmap(
        jax.vmap(encode_action, (0, None, 0, 0, None)), (0, None, 0, 0, None)
    )(rngs, action_encoder_state, actions, latent_states[:, :-1], action_encoder_params)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, latent_states.shape[0])
    losses = jax.vmap(loss_disperse_one_trajectory, (0, 0, 0))(
        rngs,
        latent_states,
        latent_actions,
    )

    return jnp.sum(losses)


def loss_condense(
    key,
    state_encoder_state,
    action_encoder_state,
    rollout_results,
    action_bounds,
    action_encoder_params=None,
):
    if action_encoder_params is None:
        action_encoder_params = action_encoder_state.params

    def one_action(key, latent_state, latent_action):
        rng, key = jax.random.split(key)
        random_action = (
            jax.random.uniform(rng, action_bounds.shape) * 2 - 1 * action_bounds
        )

        rng, key = jax.random.split(key)
        random_latent_action = encode_action(
            rng,
            action_encoder_state,
            random_action,
            latent_state,
            action_encoder_params,
        )

        diff = random_latent_action - latent_action
        bounded_diffs_mag_sq = diff.T @ diff

        return sigmoid(bounded_diffs_mag_sq) * 2 - 1

    states, actions = rollout_results

    states = states[:, :-1]
    states = rearrange(states, "r t d -> (r t) d")
    actions = rearrange(actions, "r t d -> (r t) d")

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[:-1])
    latent_states = jax.vmap(encode_state, (0, None, 0))(
        rngs, state_encoder_state, states
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, actions.shape[:-1])
    latent_actions = jax.vmap(encode_action, (0, None, 0, 0, None))(
        rngs, action_encoder_state, actions, latent_states, action_encoder_params
    )
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, latent_states.shape[0])
    result = jax.vmap(one_action, (0, 0, 0))(rngs, latent_states, latent_actions)

    return jnp.sum(result)
