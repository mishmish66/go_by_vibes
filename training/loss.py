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
    size_state_action_neighborhood,
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
    key,
    inferred_latent_states_prime_gauss_params,
    gt_next_latent_states,
):
    rng, key = jax.random.split(key)
    sampled_inferred = sample_gaussian(rng, inferred_latent_states_prime_gauss_params)

    msle = jnp.mean(jnp.log(jnp.square(sampled_inferred - gt_next_latent_states) + 1))

    return msle

    # log_probs = jax.vmap(
    #     eval_log_gaussian,
    #     (0, 0),
    # )(inferred_latent_states_prime_gauss_params, gt_next_latent_states)

    # return jnp.mean(-log_probs)


def loss_reconstruction(
    key,
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
    rng, key = jax.random.split(key)
    sampled_inferred = sample_gaussian(rng, gaussian_params)

    msle = jnp.mean(jnp.log(jnp.square(sampled_inferred - gt_value) + 1))
    return msle

    # probs = jax.vmap(
    #     eval_log_gaussian,
    #     (0, 0),
    # )(gaussian_params, gt_value)
    # return -jnp.mean(probs)


def loss_action_neighborhood_size(
    action_neighborhood_sizes,
):
    return jnp.mean(-action_neighborhood_sizes)


def loss_smoothness(
    original_latent_states,
    neighborhood_result_latent_states,
    first_unknown_next_state_i,
):
    diffs = original_latent_states - neighborhood_result_latent_states
    diff_mags = jnp.linalg.norm(diffs, ord=1, axis=-1)
    neighborhood_violation = jnp.maximum(diff_mags - 1.0, 0)
    log_neighborhood_violation = jnp.log(neighborhood_violation + 1e-6)

    # Mask out the losses from known past states
    future_state_mask = make_mask(
        original_latent_states.shape[0], first_unknown_next_state_i
    )
    sum_future_inferred_log_neighborhood_violation = jnp.sum(
        log_neighborhood_violation * future_state_mask
    )

    future_inferred_state_count = jnp.sum(future_state_mask)
    mean_future_inferred_log_neighborhood_violation = (
        sum_future_inferred_log_neighborhood_violation / future_inferred_state_count
    )

    return mean_future_inferred_log_neighborhood_violation


def loss_disperse(
    key,
    latent_states,
    state_samples,
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

    sampled_latent_states = jax.random.choice(
        rngs[0], latent_states, shape=[state_samples], replace=False
    )

    pairwise_diffs = sampled_latent_states[None] - sampled_latent_states[:, None]
    pairwise_diff_mags = jnp.linalg.norm(pairwise_diffs, ord=1, axis=-1)
    pairwise_diff_mags = pairwise_diff_mags[
        jnp.triu_indices_from(pairwise_diff_mags, k=1)
    ]

    return -jnp.mean(jnp.log(pairwise_diff_mags + 1))


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
    rng, key = jax.random.split(key)
    state_reconstruction_loss = loss_reconstruction(
        rng,
        state_space_gaussians,
        states,
    )
    rng, key = jax.random.split(key)
    action_reconstruction_loss = loss_reconstruction(
        rng,
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

    rng, key = jax.random.split(key)
    random_latent_states = jax.random.ball(
        rng,
        d=encoded_state_dim,
        p=1,
        shape=[latent_states.shape[0] // 4],
    )
    random_state_neighborhood_sizes = jax.vmap(
        jax.tree_util.Partial(
            size_state_action_neighborhood,
            vibe_state=vibe_state,
            vibe_config=train_config,
        )
    )(random_latent_states)

    state_dispersion_loss = loss_disperse(
        rngs[3],
        latent_states=latent_states,
        state_samples=8,
    )
    action_dispersion_loss = loss_disperse(
        rngs[4],
        latent_states=latent_actions,
        state_samples=8,
    )
    dispersion_loss = state_dispersion_loss + action_dispersion_loss
    result_infos = result_infos.add_loss_info("dispersion_loss", dispersion_loss)

    condensation_loss = state_condensation_loss + action_condensation_loss

    result_infos = result_infos.add_loss_info("condensation_loss", condensation_loss)

    action_neighborhood_loss = loss_action_neighborhood_size(
        random_state_neighborhood_sizes
    )
    result_infos = result_infos.add_loss_info(
        "action_neighborhood_loss", action_neighborhood_loss
    )

    return (
        Losses.init(
            reconstruction_loss=reconstruction_loss,
            condensation_loss=condensation_loss,
            dispersion_loss=dispersion_loss,
            action_neighborhood_loss=action_neighborhood_loss,
        ),
        result_infos,
    )


def composed_random_index_losses(
    key,
    states,
    actions,
    vibe_state: VibeState,
    train_config: TrainConfig,
    context_length=509,
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

    action_neighborhoods_sizes = jax.vmap(
        jax.tree_util.Partial(
            size_state_action_neighborhood,
            vibe_state=vibe_state,
            vibe_config=train_config,
        )
    )(prev_latent_states)

    # Generate a random index and give it a bunch of names
    random_index = jax.random.randint(key, [], 0, prev_latent_states.shape[0])
    last_known_prev_state_i = random_index
    current_action_i = random_index
    first_unknown_next_state_i = random_index
    # Make masks
    known_prev_state_mask = ~make_mask(
        prev_latent_states.shape[0], last_known_prev_state_i + 1
    )
    unknown_next_state_mask = make_mask(
        next_latent_states.shape[0], first_unknown_next_state_i
    )

    last_known_state = prev_latent_states[last_known_prev_state_i]

    # Now we get predict the next latent states
    latent_prime_gaussians = get_latent_state_prime_gaussians(
        last_known_state,
        latent_actions=latent_actions,
        vibe_state=vibe_state,
        vibe_config=train_config,
        current_action_i=current_action_i,
    )

    latent_prime_gaussians_inferred = einsum(
        latent_prime_gaussians,
        unknown_next_state_mask,
        "i ..., i -> i ...",
    )
    gt_next_latent_states = einsum(
        next_latent_states,
        unknown_next_state_mask,
        "i ..., i -> i ...",
    )

    # Evaluate the forward loss
    rng, key = jax.random.split(key)
    forward_loss = loss_forward(
        rng, latent_prime_gaussians_inferred, gt_next_latent_states
    )

    # Evaluate the smoothness loss
    # First we resample the indexed state
    rng, key = jax.random.split(key)
    neighborhood_latent_start_state = get_neighborhood_state(rng, last_known_state)
    # Now we resample the latent actions
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, latent_actions.shape[0])
    neighborhood_latent_actions = jax.vmap(get_neighborhood_action)(
        rngs, latent_actions, action_neighborhoods_sizes
    )

    # Now we infer forward from the neighborhood state and actions
    rng, key = jax.random.split(key)
    neighborhood_next_states_prime = infer_states(
        rng,
        latent_start_state=neighborhood_latent_start_state,
        latent_actions=neighborhood_latent_actions,
        vibe_state=vibe_state,
        vibe_config=train_config,
        current_action_i=current_action_i,
    )
    # Now we evaluate the smoothness loss
    smoothness_loss = loss_smoothness(
        original_latent_states=next_latent_states,
        neighborhood_result_latent_states=neighborhood_next_states_prime,
        first_unknown_next_state_i=first_unknown_next_state_i,
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
    action_neighborhood_loss: any

    @classmethod
    def init(
        cls,
        reconstruction_loss=0,
        forward_loss=0,
        smoothness_loss=0,
        dispersion_loss=0,
        condensation_loss=0,
        action_neighborhood_loss=0,
    ):
        return cls(
            reconstruction_loss=reconstruction_loss,
            forward_loss=forward_loss,
            smoothness_loss=smoothness_loss,
            dispersion_loss=dispersion_loss,
            condensation_loss=condensation_loss,
            action_neighborhood_loss=action_neighborhood_loss,
        )

    def tree_flatten(self):
        return [
            self.reconstruction_loss,
            self.forward_loss,
            self.smoothness_loss,
            self.dispersion_loss,
            self.condensation_loss,
            self.action_neighborhood_loss,
        ], None

    @classmethod
    def tree_unflatten(cls, aux, data):
        return cls.init(
            reconstruction_loss=data[0],
            forward_loss=data[1],
            smoothness_loss=data[2],
            dispersion_loss=data[3],
            condensation_loss=data[4],
            action_neighborhood_loss=data[5],
        )

    @classmethod
    def merge(cls, a, b):
        return cls.init(
            reconstruction_loss=a.reconstruction_loss + b.reconstruction_loss,
            forward_loss=a.forward_loss + b.forward_loss,
            smoothness_loss=a.smoothness_loss + b.smoothness_loss,
            dispersion_loss=a.dispersion_loss + b.dispersion_loss,
            condensation_loss=a.condensation_loss + b.condensation_loss,
            action_neighborhood_loss=a.action_neighborhood_loss
            + b.action_neighborhood_loss,
        )

    def scale_gate_info(self, train_config: TrainConfig):
        infos = Infos.init()

        forward_gate = make_gate_value(
            self.reconstruction_loss,
            train_config.forward_gate_sharpness,
            train_config.forward_gate_center,
        )
        smoothness_gate = (
            make_gate_value(
                self.forward_loss,
                train_config.smoothness_gate_sharpness,
                train_config.smoothness_gate_center,
            )
            * forward_gate
        )
        dispersion_gate = (
            make_gate_value(
                self.smoothness_loss,
                train_config.dispersion_gate_sharpness,
                train_config.dispersion_gate_center,
            )
            * smoothness_gate
        )
        condensation_gate = (
            make_gate_value(
                self.smoothness_loss,
                train_config.condensation_gate_sharpness,
                train_config.condensation_gate_center,
            )
            * smoothness_gate
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
        scaled_action_neighborhood_loss = (
            self.action_neighborhood_loss * train_config.action_neighborhood_weight
        )

        scaled_gated_forward_loss = scaled_forward_loss * forward_gate
        scaled_gated_smoothness_loss = scaled_smoothness_loss * smoothness_gate
        scaled_gated_dispersion_loss = scaled_dispersion_loss * dispersion_gate
        scaled_gated_condensation_loss = scaled_condensation_loss * condensation_gate
        scaled_gated_action_neighborhood_loss = (
            scaled_action_neighborhood_loss * smoothness_gate
        )

        total_loss = (
            scaled_reconstruction_loss
            + scaled_forward_loss
            + scaled_smoothness_loss
            + scaled_dispersion_loss
            + scaled_condensation_loss
            + scaled_gated_action_neighborhood_loss
        )

        infos = infos.add_loss_info("total_loss", total_loss)

        infos = infos.add_loss_info("reconstruction_loss", scaled_reconstruction_loss)
        infos = infos.add_loss_info("forward_loss", scaled_forward_loss)
        infos = infos.add_loss_info("smoothness_loss", scaled_smoothness_loss)
        infos = infos.add_loss_info("dispersion_loss", scaled_dispersion_loss)
        infos = infos.add_loss_info("condensation_loss", scaled_condensation_loss)
        infos = infos.add_loss_info(
            "action_neighborhood_loss", scaled_action_neighborhood_loss
        )

        infos = infos.add_plain_info("forward_gate", forward_gate)
        infos = infos.add_plain_info("smoothness_gate", smoothness_gate)
        infos = infos.add_plain_info("dispersion_gate", dispersion_gate)
        infos = infos.add_plain_info("condensation_gate", condensation_gate)

        result_loss = Losses.init(
            reconstruction_loss=scaled_reconstruction_loss,
            forward_loss=scaled_gated_forward_loss,
            smoothness_loss=scaled_gated_smoothness_loss,
            dispersion_loss=scaled_gated_dispersion_loss,
            condensation_loss=scaled_gated_condensation_loss,
            action_neighborhood_loss=scaled_gated_action_neighborhood_loss,
        )

        return result_loss, infos

    def to_list(self):
        return [
            self.reconstruction_loss,
            self.forward_loss,
            self.smoothness_loss,
            self.dispersion_loss,
            self.condensation_loss,
            self.action_neighborhood_loss,
        ]

    def replace(self, **kwargs):
        return Losses.init(
            reconstruction_loss=kwargs.get(
                "reconstruction_loss", self.reconstruction_loss
            ),
            forward_loss=kwargs.get("forward_loss", self.forward_loss),
            smoothness_loss=kwargs.get("smoothness_loss", self.smoothness_loss),
            dispersion_loss=kwargs.get("dispersion_loss", self.dispersion_loss),
            condensation_loss=kwargs.get("condensation_loss", self.condensation_loss),
            action_neighborhood_loss=kwargs.get(
                "action_neighborhood_loss", self.action_neighborhood_loss
            ),
        )

    @classmethod
    def from_list(cls, self):
        return cls.init(
            reconstruction_loss=self[0],
            forward_loss=self[1],
            smoothness_loss=self[2],
            dispersion_loss=self[3],
            condensation_loss=self[4],
            action_neighborhood_loss=self[5],
        )

    def total(self):
        return sum(self.to_list())


def make_gate_value(x, sharpness, center):
    sgx = jax.lax.stop_gradient(x)
    return (1 + jnp.exp(sharpness * (sgx - center))) ** (-1 / 16)
