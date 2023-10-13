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
    neighborhood_violation = jnp.maximum(diff_mags - 1.0, 0)

    return jnp.mean(neighborhood_violation)


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


def composed_whole_traj_losses(
    key,
    states,
    actions,
    vibe_state: VibeState,
    train_config: TrainConfig,
):
    """This will sample one encoding from the encoder and evaluate losses"""

    result_infos = Infos.init()

    latent_state_gauss_params = jax.vmap(get_latent_state_gaussian, (0, None, None))(
        states, vibe_state, train_config
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[0])
    latent_states = jax.vmap(sample_gaussian)(rngs, latent_state_gauss_params)

    prev_latent_states = latent_states[:-1]

    latent_action_gauss_params = jax.vmap(
        get_latent_action_gaussian,
        (0, 0, None, None),
    )(actions, prev_latent_states, vibe_state, train_config)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, actions.shape[0])
    latent_actions = jax.vmap(sample_gaussian)(rngs, latent_action_gauss_params)

    state_space_gaussians = jax.vmap(
        get_state_space_gaussian,
        in_axes=(0, None, None),
    )(latent_states, vibe_state, train_config)
    action_space_gaussians = jax.vmap(
        get_action_space_gaussian,
        in_axes=(0, 0, None, None),
    )(latent_actions, prev_latent_states, vibe_state, train_config)

    # Evaluate reconstruction loss:
    reconstruction_loss = loss_reconstruction(
        state_space_gaussians,
        action_space_gaussians,
        states,
        actions,
    )

    result_infos = result_infos.add_loss_info(
        "reconstruction_loss", reconstruction_loss
    )

    # Evaluate dispersion loss
    dispersion_loss = loss_disperse(latent_states)

    # Evaluate condensation loss
    # Resample actions from action space
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, latent_states.shape[0])
    random_actions = jax.vmap(train_config.env_config.random_action)(rngs)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, random_actions.shape[0])
    latent_random_actions = jax.vmap(encode_action, (0, 0, 0, None, None))(
        rngs, random_actions, latent_states, vibe_state, train_config
    )

    condensation_loss = loss_condense(latent_random_actions)

    reconstructed_state_vars = state_space_gaussians[
        ..., train_config.env_config.state_dim :
    ]
    reconstructed_state_means = state_space_gaussians[
        ..., : train_config.env_config.state_dim
    ]
    reconstructed_state_mean_stdev_log = jnp.log(
        jnp.mean(jnp.std(reconstructed_state_means, axis=0))
    )

    reconstructed_action_vars = action_space_gaussians[
        ..., train_config.env_config.act_dim :
    ]
    reconstructed_action_means = action_space_gaussians[
        ..., : train_config.env_config.act_dim
    ]
    reconstructed_action_mean_stdev_log = jnp.log(
        jnp.mean(jnp.std(reconstructed_action_means, axis=0))
    )

    latent_state_variances = latent_state_gauss_params[..., encoded_state_dim:]
    latent_state_means = latent_state_gauss_params[..., :encoded_state_dim]
    latent_state_mean_stdev_log = jnp.log(jnp.mean(jnp.std(latent_state_means, axis=0)))

    latent_action_variances = latent_action_gauss_params[..., encoded_action_dim:]
    latent_action_means = latent_action_gauss_params[..., :encoded_action_dim]
    latent_action_mean_stdev_log = jnp.log(
        jnp.mean(jnp.std(latent_action_means, axis=0))
    )

    result_infos = result_infos.add_loss_info("dispersion_loss", dispersion_loss)
    result_infos = result_infos.add_loss_info("condensation_loss", condensation_loss)

    result_infos = result_infos.add_plain_info(
        "latent_state_var_logs", jnp.log(latent_state_variances)
    )
    result_infos = result_infos.add_plain_info(
        "latent_state_mean_stdev_log", latent_state_mean_stdev_log
    )
    result_infos = result_infos.add_plain_info(
        "latent_action_var_logs", jnp.log(latent_action_variances)
    )
    result_infos = result_infos.add_plain_info(
        "latent_action_mean_stdev_log", latent_action_mean_stdev_log
    )

    result_infos = result_infos.add_plain_info(
        "reconstructed_state_var_logs", jnp.log(reconstructed_state_vars)
    )
    result_infos = result_infos.add_plain_info(
        "reconstructed_state_mean_stdev_log", reconstructed_state_mean_stdev_log
    )
    result_infos = result_infos.add_plain_info(
        "reconstructed_action_var_logs", jnp.log(reconstructed_action_vars)
    )
    result_infos = result_infos.add_plain_info(
        "reconstructed_action_mean_stdev_log", reconstructed_action_mean_stdev_log
    )

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
        inferred_latent_states_prime_sampled, gt_next_latent_states
    )

    state_prime_diffs = inferred_latent_states_prime_sampled - gt_next_latent_states

    state_prime_diff_mags = jnp.linalg.norm(state_prime_diffs, axis=-1)
    state_prime_diff_mag_logs = jnp.log(state_prime_diff_mags)

    result_infos = result_infos.add_masked_info(
        "state_prime_diff_mag_logs",
        state_prime_diff_mag_logs,
        inferred_next_state_mask,
    )

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

    def get_total(self, train_config: TrainConfig):
        infos = Infos.init()

        forward_gate = make_gate_value(
            self.reconstruction_loss,
            train_config.forward_gate_sharpness,
            train_config.forward_gate_center,
        )
        smoothness_gate = make_gate_value(
            self.reconstruction_loss,
            train_config.smoothness_gate_sharpness,
            train_config.smoothness_gate_center,
        )
        dispersion_gate = make_gate_value(
            self.reconstruction_loss,
            train_config.dispersion_gate_sharpness,
            train_config.dispersion_gate_center,
        )
        condensation_gate = make_gate_value(
            self.reconstruction_loss,
            train_config.condensation_gate_sharpness,
            train_config.condensation_gate_center,
        )

        scaled_reconstruction_loss = (
            self.reconstruction_loss * train_config.reconstruction_weight
        )
        scaled_forward_loss = (
            self.forward_loss * train_config.forward_weight * forward_gate
        )
        scaled_smoothness_loss = (
            self.smoothness_loss * train_config.smoothness_weight * smoothness_gate
        )
        scaled_dispersion_loss = (
            self.dispersion_loss * train_config.dispersion_weight * dispersion_gate
        )
        scaled_condensation_loss = (
            self.condensation_loss
            * train_config.condensation_weight
            * condensation_gate
        )

        total_loss = (
            scaled_reconstruction_loss
            + scaled_forward_loss
            + scaled_smoothness_loss
            + scaled_dispersion_loss
            + scaled_condensation_loss
        )

        infos.add_loss_info("total_loss", total_loss)

        infos.add_loss_info("reconstruction_loss", scaled_reconstruction_loss)
        infos.add_loss_info("forward_loss", scaled_forward_loss)
        infos.add_loss_info("smoothness_loss", scaled_smoothness_loss)
        infos.add_loss_info("dispersion_loss", scaled_dispersion_loss)
        infos.add_loss_info("condensation_loss", scaled_condensation_loss)

        infos.add_plain_info("forward_gate", forward_gate)
        infos.add_plain_info("smoothness_gate", smoothness_gate)
        infos.add_plain_info("dispersion_gate", dispersion_gate)
        infos.add_plain_info("condensation_gate", condensation_gate)

        return total_loss, infos


def make_gate_value(x, sharpness, center):
    return jax.lax.stop_gradient(1 / (1 + jnp.exp(sharpness * (x - center))))
