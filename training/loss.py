import jax
from jax import numpy as jnp
from jax.scipy.stats.multivariate_normal import pdf as multinorm_pdf
from jax.scipy.stats.multivariate_normal import logpdf as multinorm_logpdf
from jax.nn import sigmoid

from jax.tree_util import register_pytree_node_class

from einops import einsum, rearrange

from .rollout import collect_rollout
from .nets import (
    encoded_state_dim,
    encoded_action_dim,
    make_inds,
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

from dataclasses import dataclass


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

    diff_norms = jnp.linalg.norm(diffs, ord=1, axis=-1)
    log_diff_norms = jnp.log(diff_norms + 1e-6)

    return jnp.mean(log_diff_norms)


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
    forward_loss_num_random_indices=16,
):
    latent_state_gaussian_params = jax.vmap(
        get_latent_state_gaussian,
        (0, None, None),
    )(
        states,
        vibe_state,
        train_config,
    )

    """ This will sample one encoding from the encoder and evaluate losses """
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

    # Evaluate forward loss:
    # Sample latent states and actions
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

    # Infer next latent states
    def per_random_index(key):
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
        last_known_state_i = random_i
        last_known_state = prev_latent_states[last_known_state_i]

        inferred_next_state_mask = make_mask(
            next_latent_states.shape[0], last_known_state_i
        )

        # Now we get predict the next latent states
        latent_states_prime_gaussians = get_latent_state_prime_gaussians(
            last_known_state,
            latent_actions,
            vibe_state,
            train_config,
            last_known_state_i,
        )

        gt_next_latent_states = einsum(
            next_latent_states,
            inferred_next_state_mask,
            "t d, t -> t d",
        )

        # Sample the next latent states
        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, latent_states_prime_gaussians.shape[0])
        latent_states_prime_sampled = jax.vmap(sample_gaussian, (0, 0))(
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
        
        # Evaluate smoothness loss
        # smoothness_loss = loss_smoothness(
            

        return forward_loss

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, forward_loss_num_random_indices)
    forward_loss_per_random_indices = jax.vmap(per_random_index)(rngs)
    forward_loss = jnp.mean(forward_loss_per_random_indices, axis=0)

    # Evaluate smoothness loss:
    # neighborhood_

    return Losses.init(
        reconstruction_loss=reconstruction_loss,
        forward_loss=forward_loss,
    )


@register_pytree_node_class
@dataclass
class Losses:
    reconstruction_loss: any
    forward_loss: any

    @classmethod
    def init(cls, reconstruction_loss, forward_loss):
        return cls(
            reconstruction_loss=reconstruction_loss,
            forward_loss=forward_loss,
        )

    def tree_flatten(self):
        return [self.reconstruction_loss, self.forward_loss], None

    @classmethod
    def tree_unflatten(cls, aux, data):
        return cls(
            reconstruction_loss=data[0],
            forward_loss=data[1],
        )

    def make_dict(self):
        return {
            "reconstruction_loss": self.reconstruction_loss,
            "forward_loss": self.forward_loss,
        }
