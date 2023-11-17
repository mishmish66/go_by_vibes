import jax
from jax import numpy as jnp
from jax.scipy.stats.multivariate_normal import pdf as multinorm_pdf
from jax.scipy.stats.multivariate_normal import logpdf as multinorm_logpdf
from jax.nn import sigmoid

from jax.tree_util import register_pytree_node_class

from einops import einsum, rearrange

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
    encode_state,
    encode_action,
    infer_states,
    get_neighborhood_state,
    get_neighborhood_action,
)

from .infos import Infos

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
    inferred_latent_states_prime_gauss_params,
    gt_next_latent_states,
    inferred_state_mask,
):
    log_probs = jax.vmap(
        eval_log_gaussian,
        (0, 0),
    )(inferred_latent_states_prime_gauss_params, gt_next_latent_states)

    relevant_log_probs = log_probs * inferred_state_mask

    return jnp.mean(-relevant_log_probs)


def loss_reconstruction(
    gaussian_params,
    gt_value,
):
    """Evaluates the probabilities of the gt_value under that gaussian.

    Args:
        gaussian_params (Array): An (n x 2d) array of n gaussian parameters with dimensionality d.
        gt_value (Array): An (n x d) array of n ground truth values with dimensionality d.

    Returns:
        Scalar: The mean negative log value of the gaussian at the gt points.
    """
    probs = jax.vmap(
        eval_log_gaussian,
        (0, 0),
    )(gaussian_params, gt_value)
    return -jnp.mean(probs)


def loss_smoothness(
    original_latent_states,
    neighborhood_result_latent_states,
):
    diffs = original_latent_states - neighborhood_result_latent_states
    diff_mags = jnp.linalg.norm(diffs, ord=1, axis=-1)
    neighborhood_violation = jnp.maximum(diff_mags - 1.0, 0)

    return jnp.mean(jnp.log(neighborhood_violation + 1e-6))


def loss_disperse(
    key,
    latent_states,
    vibe_state: VibeState,
    train_config: TrainConfig,
    samples=16,
):
    """This function computes the dispersion loss between a sampled set of latents.

    Args:
        key (PRNGKey): The rng key to sample with
        latent_states (Array): An (n x d_s) array of n latent states with dimensionality d_s
        latent_actions (Array): An (n x d_a) array of n latent actions with dimensionality d_a
        samples (int, optional): The number of samples to take. Defaults to 16.

    Returns:
        Scalar: The mean loss value.
    """

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, 2)

    sampled_latent_start_states = jax.random.choice(
        rngs[0], latent_states, shape=[samples], replace=True
    )
    uniformly_sampled_action_pairs = jax.random.ball(
        rngs[1], d=encoded_action_dim, p=1, shape=[samples, 2]
    )

    def infer_pair(key, start_state, action_pair):
        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, 2)

        next_state_pair = jax.vmap(infer_states, in_axes=[0, None, 0, None, None])(
            rngs,
            start_state,
            action_pair[..., None, :],
            vibe_state,
            train_config,
        )
        return next_state_pair[..., 0, :]

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, samples)
    successor_pairs = jax.vmap(infer_pair)(
        rngs, sampled_latent_start_states, uniformly_sampled_action_pairs
    )

    action_pair_dists = jnp.linalg.norm(
        uniformly_sampled_action_pairs[..., 0, :]
        - uniformly_sampled_action_pairs[..., 1, :],
        axis=-1,
        ord=1,
    )
    successor_pair_dists = jnp.linalg.norm(
        successor_pairs[..., 0] - successor_pairs[..., 1],
        axis=-1,
        ord=1,
    )

    dist_diff_mags = jnp.abs(successor_pair_dists - action_pair_dists)

    diff_norms = jnp.linalg.norm(dist_diff_mags, ord=1, axis=-1)
    diff_norm_logs = jnp.log(diff_norms + 1e-6)

    return jnp.mean(diff_norm_logs)


def loss_condense(
    key,
    latents,
    target_radius=1.0,
    samples=16,
):
    """This function samples a few latent and computes the radius violation of that one latent.

    Args:
        key (PRNGKey): The rng key to use.
        latents (Array): An (n x d) array of n latent variables with dimensionality d.
        target_radius (float, optional): The radius to condense to. Defaults to 1.0.

    Returns:
        Scalar: The loss value for the sampled latent.
    """
    rng, key = jax.random.split(key)
    latents = jax.random.choice(rng, latents, shape=[samples], replace=False)
    latent_action_rads = jnp.linalg.norm(latents, ord=1, axis=-1)

    latent_action_rad_violations = jnp.maximum(latent_action_rads - target_radius, 0)

    return jnp.mean(jnp.log(latent_action_rad_violations + 1e-6))


def unordered_losses(
    key,
    states,
    actions,
    vibe_state: VibeState,
    train_config: TrainConfig,
):
    """This will sample one encoding from the encoder and evaluate the losses that require a whole trajectory.

    Args:
        key (PRNGKey): The random key to use.
        states (Array): An (n x d) array of n states with dimensionality d.
        actions (Array): An (n x d) array of n actions with dimensionality d.
        vibe_state (VibeState): The state of the network.
        train_config (TrainConfig): The training configuration.

    Returns:
        (Losses, Info): A tuple of the loss and info objects.
    """

    result_infos = Infos.init()

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[0])
    latent_states = jax.vmap(
        encode_state,
        (0, 0, None, None),
    )(rngs, states, vibe_state, train_config)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, actions.shape[0])
    latent_actions = jax.vmap(
        encode_action,
        (0, 0, 0, None, None),
    )(rngs, actions, latent_states, vibe_state, train_config)

    state_space_gaussians = jax.vmap(
        get_state_space_gaussian,
        in_axes=(0, None, None),
    )(latent_states, vibe_state, train_config)
    action_space_gaussians = jax.vmap(
        get_action_space_gaussian,
        in_axes=(0, 0, None, None),
    )(latent_actions, latent_states, vibe_state, train_config)

    # Evaluate reconstruction loss:
    state_reconstruction_loss = loss_reconstruction(
        state_space_gaussians,
        states,
    )
    action_reconstruction_loss = loss_reconstruction(
        action_space_gaussians,
        actions,
    )
    reconstruction_loss = state_reconstruction_loss + action_reconstruction_loss

    result_infos = result_infos.add_loss_info(
        "reconstruction_loss", reconstruction_loss
    )

    # Evaluate dispersion loss
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, 4)

    state_condensation_loss = loss_condense(
        rngs[1],
        latent_states,
        samples=latent_states.shape[0],
        target_radius=train_config.state_radius,
    )
    action_condensation_loss = loss_condense(
        rngs[2],
        latent_actions,
        samples=latent_actions.shape[0],
        target_radius=train_config.action_radius,
    )

    dispersion_loss = loss_disperse(
        rngs[3], latent_states, vibe_state, train_config, samples=128
    )
    condensation_loss = state_condensation_loss + action_condensation_loss

    result_infos = result_infos.add_loss_info("dispersion_loss", dispersion_loss)
    result_infos = result_infos.add_loss_info("condensation_loss", condensation_loss)

    return (
        Losses.init(
            reconstruction_loss=reconstruction_loss,
            dispersion_loss=dispersion_loss,
            condensation_loss=condensation_loss,
        ),
        result_infos,
    )


def composed_random_index_losses(
    key,
    states,
    actions,
    vibe_state: VibeState,
    train_config: TrainConfig,
):
    result_infos = Infos.init()

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[0])
    latent_states = jax.vmap(encode_state, (0, 0, None, None))(
        rngs, states, vibe_state, train_config
    )

    prev_latent_states = latent_states[:-1]
    next_latent_states = latent_states[1:]

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, actions.shape[0])
    latent_actions = jax.vmap(encode_action, (0, 0, 0, None, None))(
        rngs, actions, prev_latent_states, vibe_state, train_config
    )

    # Generate a random index to make the network guess
    # the second part of the latent state sequence
    # give it some padding so it has to predict at least
    # an eighth of the sequence
    end_padding = prev_latent_states.shape[0] // 8
    random_i = jax.random.randint(key, [], 0, prev_latent_states.shape[0] - end_padding)

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
        latent_states_prime_gaussians, gt_next_latent_states, inferred_next_state_mask
    )

    state_prime_diffs = inferred_latent_states_prime_sampled - gt_next_latent_states

    state_prime_diff_mags = jnp.linalg.norm(state_prime_diffs, axis=-1)
    state_prime_diff_mag_logs = jnp.log(state_prime_diff_mags)

    # result_infos = result_infos.add_masked_info(
    #     "state_prime_diff_mag_logs",
    #     state_prime_diff_mag_logs,
    #     inferred_next_state_mask,
    # )

    # Evaluate the smoothness loss
    # First we resample the indexed state
    rng, key = jax.random.split(key)
    neighborhood_latent_start_state = get_neighborhood_state(
        rng,
        prev_latent_states[last_known_state_i],
    )
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, latent_actions.shape[0])
    neighborhood_latent_actions = jax.vmap(get_neighborhood_action)(
        rngs, latent_actions
    )

    # Now we resample the latent actions
    # Now we model forward from that state
    rng, key = jax.random.split(key)
    neighborhood_next_states_prime = infer_states(
        rng,
        neighborhood_latent_start_state,
        neighborhood_latent_actions,
        vibe_state,
        train_config,
        last_known_state_i,
    )
    # Now we evaluate the smoothness loss
    smoothness_loss = loss_smoothness(
        next_latent_states,
        neighborhood_next_states_prime,
    )

    result_infos = result_infos.add_loss_info("smoothness_loss", smoothness_loss)
    result_infos = result_infos.add_loss_info("forward_loss", forward_loss)

    return (
        Losses.init(forward_loss=forward_loss, smoothness_loss=smoothness_loss),
        result_infos,
    )


@register_pytree_node_class
@dataclass
class Losses:
    reconstruction_loss: any
    forward_loss: any
    smoothness_loss: any
    dispersion_loss: any
    condensation_loss: any

    @classmethod
    def init(
        cls,
        reconstruction_loss=0,
        forward_loss=0,
        smoothness_loss=0,
        dispersion_loss=0,
        condensation_loss=0,
    ):
        return cls(
            reconstruction_loss=reconstruction_loss,
            forward_loss=forward_loss,
            smoothness_loss=smoothness_loss,
            dispersion_loss=dispersion_loss,
            condensation_loss=condensation_loss,
        )

    def tree_flatten(self):
        return [
            self.reconstruction_loss,
            self.forward_loss,
            self.smoothness_loss,
            self.dispersion_loss,
            self.condensation_loss,
        ], None

    @classmethod
    def tree_unflatten(cls, aux, data):
        return cls.init(
            reconstruction_loss=data[0],
            forward_loss=data[1],
            smoothness_loss=data[2],
            dispersion_loss=data[3],
            condensation_loss=data[4],
        )

    @classmethod
    def merge(cls, a, b):
        return cls.init(
            reconstruction_loss=a.reconstruction_loss + b.reconstruction_loss,
            forward_loss=a.forward_loss + b.forward_loss,
            smoothness_loss=a.smoothness_loss + b.smoothness_loss,
            dispersion_loss=a.dispersion_loss + b.dispersion_loss,
            condensation_loss=a.condensation_loss + b.condensation_loss,
        )

    def scale_gate_info(self, train_config: TrainConfig):
        infos = Infos.init()

        inverse_reconstruction_gate = 1 - make_gate_value(
            self.reconstruction_loss,
            train_config.inverse_reconstruction_gate_sharpness,
            train_config.inverse_reconstruction_gate_center,
        )

        inverse_forward_gate = 1 - make_gate_value(
            self.forward_loss,
            train_config.inverse_forward_gate_sharpness,
            train_config.inverse_forward_gate_center,
        )

        forward_gate = make_gate_value(
            self.reconstruction_loss,
            train_config.forward_gate_sharpness,
            train_config.forward_gate_center,
        )
        smoothness_gate = make_gate_value(
            self.forward_loss,
            train_config.smoothness_gate_sharpness,
            train_config.smoothness_gate_center,
        )
        dispersion_gate = make_gate_value(
            self.smoothness_loss,
            train_config.dispersion_gate_sharpness,
            train_config.dispersion_gate_center,
        )
        condensation_gate = make_gate_value(
            self.smoothness_loss,
            train_config.condensation_gate_sharpness,
            train_config.condensation_gate_center,
        )

        scaled_reconstruction_loss = (
            self.reconstruction_loss * train_config.reconstruction_weight
        )
        scaled_forward_loss = self.forward_loss * train_config.forward_weight
        scaled_smoothness_loss = self.smoothness_loss * train_config.smoothness_weight
        scaled_dispersion_loss = self.dispersion_loss * train_config.dispersion_weight
        scaled_condensation_loss = (
            self.condensation_loss * train_config.condensation_weight
        )

        scaled_gated_reconstruction_loss = (
            scaled_reconstruction_loss * inverse_reconstruction_gate
        )
        scaled_gated_forward_loss = (
            scaled_forward_loss * forward_gate * inverse_forward_gate
        )
        scaled_gated_smoothness_loss = scaled_smoothness_loss * smoothness_gate
        scaled_gated_dispersion_loss = scaled_dispersion_loss * dispersion_gate
        scaled_gated_condensation_loss = scaled_condensation_loss * condensation_gate

        total_loss = (
            scaled_reconstruction_loss
            + scaled_forward_loss
            + scaled_smoothness_loss
            + scaled_dispersion_loss
            + scaled_condensation_loss
        )

        infos = infos.add_loss_info("total_loss", total_loss)

        infos = infos.add_loss_info("reconstruction_loss", scaled_reconstruction_loss)
        infos = infos.add_loss_info("forward_loss", scaled_forward_loss)
        infos = infos.add_loss_info("smoothness_loss", scaled_smoothness_loss)
        infos = infos.add_loss_info("dispersion_loss", scaled_dispersion_loss)
        infos = infos.add_loss_info("condensation_loss", scaled_condensation_loss)

        infos = infos.add_plain_info(
            "inverse_reconstruction_gate", inverse_reconstruction_gate
        )
        infos = infos.add_plain_info("inverse_forward_gate", inverse_forward_gate)
        infos = infos.add_plain_info("forward_gate", forward_gate)
        infos = infos.add_plain_info("smoothness_gate", smoothness_gate)
        infos = infos.add_plain_info("dispersion_gate", dispersion_gate)
        infos = infos.add_plain_info("condensation_gate", condensation_gate)

        result_loss = Losses.init(
            reconstruction_loss=scaled_gated_reconstruction_loss,
            forward_loss=scaled_gated_forward_loss,
            smoothness_loss=scaled_gated_smoothness_loss,
            dispersion_loss=scaled_gated_dispersion_loss,
            condensation_loss=scaled_gated_condensation_loss,
        )

        return result_loss, infos

    def to_list(self):
        return [
            self.reconstruction_loss,
            self.forward_loss,
            self.smoothness_loss,
            self.dispersion_loss,
            self.condensation_loss,
        ]

    def total(self):
        return sum(self.to_list())


def make_gate_value(x, sharpness, center):
    return jax.lax.stop_gradient(1 / (1 + jnp.exp(sharpness * (x - center))))
