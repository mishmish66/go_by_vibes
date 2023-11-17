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
):
    log_probs = jax.vmap(
        eval_log_gaussian,
        (0, 0),
    )(inferred_latent_states_prime_gauss_params, gt_next_latent_states)

    return jnp.mean(-log_probs)


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
    start_state_samples=16,
    random_action_samples=16,
):
    """This function computes the dispersion loss between a sampled set of latents.

    Args:
        key (PRNGKey): The rng key to sample with
        latent_states (Array): An (n x d_s) array of n latent states with dimensionality d_s
        samples (int, optional): The number of samples to take. Defaults to 16.

    Returns:
        Scalar: The mean loss value.
    """

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, 2)

    sampled_latent_start_states = jax.random.choice(
        rngs[0], latent_states, shape=[start_state_samples], replace=False
    )

    uniformly_sampled_action_groups = jax.random.ball(
        rngs[1],
        d=encoded_action_dim,
        p=1,
        shape=[start_state_samples, random_action_samples],
    )

    def do_group(key, latent_start_state):
        rng, key = jax.random.split(key)
        rngs = jax.random.split(rng, random_action_samples)
        uniformly_sampled_action_group = jax.random.ball(
            rng,
            d=encoded_action_dim,
            p=1,
            shape=[random_action_samples],
        )

        next_states = jax.vmap(
            jax.tree_util.Partial(
                infer_states,
                latent_start_state=latent_start_state,
                vibe_state=vibe_state,
                vibe_config=train_config,
            )
        )(
            rngs,
            latent_actions=uniformly_sampled_action_group[..., None, :],
        ).squeeze(
            -2
        )

        pairwise_action_dists = jnp.linalg.norm(
            uniformly_sampled_action_group[..., None]
            - uniformly_sampled_action_group[..., None, :],
            axis=-1,
            ord=1,
        )
        pairwise_action_dists = pairwise_action_dists[
            jnp.triu_indices_from(pairwise_action_dists, k=1)
        ]

        pairwise_successor_dists = jnp.linalg.norm(
            next_states[..., None] - next_states[..., None, :], axis=-1, ord=1
        )
        pairwise_successor_dists = pairwise_successor_dists[
            jnp.triu_indices_from(pairwise_successor_dists, k=1)
        ]

        return jnp.abs(pairwise_action_dists - pairwise_successor_dists)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, start_state_samples)
    per_start_state_dist_diff_mags = jax.vmap(do_group)(
        rngs, sampled_latent_start_states
    )

    dist_diff_mags = jnp.ravel(per_start_state_dist_diff_mags)

    # jax.debug.print("action_pair_dists: {}", action_pair_dists)
    # jax.debug.print("successor_pair_dists: {}", successor_pair_dists)

    diff_norm_logs = jnp.log(dist_diff_mags + 1e-6)

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
        rngs[3],
        latent_states,
        vibe_state,
        train_config,
        start_state_samples=latent_states.shape[0],
    )
    result_infos = result_infos.add_loss_info("dispersion_loss", dispersion_loss)

    condensation_loss = state_condensation_loss + action_condensation_loss

    result_infos = result_infos.add_loss_info("condensation_loss", condensation_loss)

    return (
        Losses.init(
            reconstruction_loss=reconstruction_loss,
            condensation_loss=condensation_loss,
            dispersion_loss=dispersion_loss,
        ),
        result_infos,
    )


def composed_random_index_losses(
    key,
    states,
    actions,
    vibe_state: VibeState,
    train_config: TrainConfig,
    context_length=128,
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

    # Generate a random index that is not in the last context_length
    slice_begin = jax.random.randint(key, [], 0, states.shape[0] - 1 - context_length)

    # Now we slice out the context
    slice_prev_latent_states = jax.lax.dynamic_slice_in_dim(
        prev_latent_states, slice_begin, context_length
    )
    slice_next_latent_states = jax.lax.dynamic_slice_in_dim(
        next_latent_states, slice_begin, context_length
    )

    slice_latent_actions = jax.lax.dynamic_slice_in_dim(
        latent_actions, slice_begin, context_length
    )

    slice_latent_start_state = slice_prev_latent_states[0]

    # Now we get predict the next latent states
    slice_latent_state_prime_gaussians = get_latent_state_prime_gaussians(
        slice_latent_start_state,
        slice_latent_actions,
        vibe_state,
        train_config,
    )

    # Evaluate the forward loss
    forward_loss = loss_forward(
        slice_latent_state_prime_gaussians, slice_next_latent_states
    )

    # Now lets predict a bunch of single state next latent states
    num_random_samples = 32
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, num_random_samples)
    random_indices = jax.random.randint(
        rng, [num_random_samples], 0, slice_prev_latent_states.shape[0] - 1
    )
    random_prev_latent_states = slice_prev_latent_states[random_indices]
    random_latent_actions = slice_latent_actions[random_indices]
    random_next_latent_states = slice_next_latent_states[random_indices]

    random_state_prime_gaussians = jax.vmap(
        get_latent_state_prime_gaussians, in_axes=(0, 0, None, None)
    )(
        random_prev_latent_states,
        random_latent_actions[:, None, :],
        vibe_state,
        train_config,
    )

    # Now we evaluate the single step forward loss
    single_step_forward_loss = loss_forward(
        random_state_prime_gaussians[..., 0, :], random_next_latent_states
    )

    forward_loss = forward_loss + single_step_forward_loss

    # Evaluate the smoothness loss
    # First we resample the indexed state
    rng, key = jax.random.split(key)
    neighborhood_latent_start_state = get_neighborhood_state(
        rng,
        slice_latent_start_state,
    )
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, slice_latent_actions.shape[0])
    neighborhood_latent_actions = jax.vmap(get_neighborhood_action)(
        rngs, slice_latent_actions
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
    )
    # Now we evaluate the smoothness loss
    smoothness_loss = loss_smoothness(
        slice_next_latent_states,
        neighborhood_next_states_prime,
    )

    result_infos = result_infos.add_loss_info("smoothness_loss", smoothness_loss)
    result_infos = result_infos.add_loss_info("forward_loss", forward_loss)

    return (
        Losses.init(
            forward_loss=forward_loss,
            smoothness_loss=smoothness_loss,
        ),
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
        ) * forward_gate
        dispersion_gate = make_gate_value(
            self.smoothness_loss,
            train_config.dispersion_gate_sharpness,
            train_config.dispersion_gate_center,
        ) * smoothness_gate
        condensation_gate = make_gate_value(
            self.smoothness_loss,
            train_config.condensation_gate_sharpness,
            train_config.condensation_gate_center,
        ) * smoothness_gate

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
