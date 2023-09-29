import jax
import jax.numpy as jnp

from clu import metrics

from jax.scipy.stats.multivariate_normal import pdf as multinorm_pdf

import flax
from flax import linen as nn
from flax.training import train_state
from flax import struct

import optax

from einops import einsum, rearrange, reduce

from physics import step
from nets import encoded_state_dim, encoded_action_dim, infer_states
from loss import (
    loss_forward,
    loss_reconstruction,
    loss_smoothness,
    loss_condense,
    loss_disperse,
    sample_gaussian,
)
from rollout import collect_rollout


def collect_latent_rollout(
    key,
    transition_model_state,
    latent_z_0,
    latent_actions,
    dt,
):
    """Collect a rollout using the latent transitions model and predefined actions."""

    rng, key = jax.random.split(key)
    net_out = infer_states(
        rng,
        transition_model_state,
        latent_z_0[None],
        latent_actions,
        dt,
    )

    return net_out


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, dims, rng, learning_rate, other={}):
    """Creates an initial `TrainState`."""
    dummy = [jnp.ones(dim) for dim in dims]
    params = module.init(rng, *dummy, **other)[
        "params"
    ]  # initialize parameters by passing a template image
    tx = optax.lion(learning_rate)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


def state_encoder_loss(
    key,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    action_bounds,
    dt,
    state_encoder_params,
    loss_weights=[1.0, 1.0, 1.0, 1.0],  # [1e6, 1.0, 1e3, 1e-6],
):
    rng, key = jax.random.split(key)
    forward_loss = loss_weights[0] * loss_forward(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        dt,
        state_encoder_params=state_encoder_params,
    )

    rng, key = jax.random.split(key)
    reconstruction_loss = loss_weights[1] * loss_reconstruction(
        rng,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        state_encoder_params=state_encoder_params,
    )

    rng, key = jax.random.split(key)
    smoothness_loss = loss_weights[2] * loss_smoothness(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        dt,
        state_encoder_params=state_encoder_params,
    )

    rng, key = jax.random.split(key)
    dispersion_loss = loss_weights[3] * loss_disperse(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        action_bounds,
        dt,
        state_encoder_params=state_encoder_params,
    )

    jax.debug.print(
        "State Encoder Losses:\n"
        + "\tForward: {}\n"
        + "\tReconstruction: {}\n"
        + "\tSmoothness: {}\n"
        + "\tDispersion: {}",
        forward_loss,
        reconstruction_loss,
        smoothness_loss,
        dispersion_loss,
    )

    return forward_loss + reconstruction_loss + smoothness_loss + dispersion_loss


def action_encoder_loss(
    key,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    action_bounds,
    dt,
    action_encoder_params,
):
    rng, key = jax.random.split(key)
    forward_loss = loss_forward(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        dt,
        action_encoder_params=action_encoder_params,
    )

    rng, key = jax.random.split(key)
    reconstruction_loss = loss_reconstruction(
        key,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        action_encoder_params=action_encoder_params,
    )

    rng, key = jax.random.split(key)
    smoothness_loss = loss_smoothness(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        dt,
        action_encoder_params=action_encoder_params,
    )

    rng, key = jax.random.split(key)
    dispersion_loss = loss_disperse(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        action_bounds,
        dt,
        action_encoder_params=action_encoder_params,
    )

    rng, key = jax.random.split(key)
    condensation_loss = loss_condense(
        rng,
        state_encoder_state,
        action_encoder_state,
        rollout_result,
        action_bounds,
        action_encoder_params,
    )

    jax.debug.print(
        "Action Encoder Losses:\n"
        + "\tForward: {}\n"
        + "\tReconstruction: {}\n"
        + "\tSmoothness: {}\n"
        + "\tDispersion: {}\n"
        + "\tCondensation: {}",
        forward_loss,
        reconstruction_loss,
        smoothness_loss,
        dispersion_loss,
        condensation_loss,
    )

    return forward_loss + reconstruction_loss + smoothness_loss + condensation_loss


def transition_model_loss(
    key,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    rollout_result,
    dt,
    transition_model_params,
):
    rng, key = jax.random.split(key)
    forward_loss = loss_forward(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        dt,
        transition_params=transition_model_params,
    )

    return forward_loss


def state_decoder_loss(
    key,
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    state_decoder_params,
):
    rng, key = jax.random.split(key)
    reconstruction_loss = loss_reconstruction(
        rng,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        state_decoder_params=state_decoder_params,
    )

    return reconstruction_loss


def action_decoder_loss(
    key,
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    action_decoder_params,
):
    rng, key = jax.random.split(key)
    reconstruction_loss = loss_reconstruction(
        rng,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        action_decoder_params=action_decoder_params,
    )

    return reconstruction_loss


def get_grads(
    key,
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    transition_model_state,
    rollout_result,
    action_bounds,
    dt,
):
    rng, key = jax.random.split(key)

    def state_encoder_loss_for_grad(state_encoder_params, key):
        rng, key = jax.random.split(key)
        return state_encoder_loss(
            rng,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
            rollout_result,
            action_bounds,
            dt,
            state_encoder_params,
        )

    def action_encoder_loss_for_grad(action_encoder_params, key):
        rng, key = jax.random.split(key)
        return action_encoder_loss(
            rng,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
            rollout_result,
            action_bounds,
            dt,
            action_encoder_params,
        )

    def transition_model_loss_for_grad(transition_model_params, key):
        rng, key = jax.random.split(key)
        return transition_model_loss(
            rng,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            rollout_result,
            dt,
            transition_model_params,
        )

    def state_decoder_loss_for_grad(state_decoder_params, key):
        rng, key = jax.random.split(key)
        return state_decoder_loss(
            rng,
            state_encoder_state,
            action_encoder_state,
            state_decoder_state,
            action_decoder_state,
            rollout_result,
            state_decoder_params,
        )

    def action_decoder_loss_for_grad(action_decoder_params, key):
        rng, key = jax.random.split(key)
        return action_decoder_loss(
            rng,
            state_encoder_state,
            action_encoder_state,
            state_decoder_state,
            action_decoder_state,
            rollout_result,
            action_decoder_params,
        )

    action_encoder_grad_fn = jax.grad(action_encoder_loss_for_grad)
    state_encoder_grad_fn = jax.grad(state_encoder_loss_for_grad)
    transition_model_grad_fn = jax.grad(transition_model_loss_for_grad)
    state_decoder_grad_fn = jax.grad(state_decoder_loss_for_grad)
    action_decoder_grad_fn = jax.grad(action_decoder_loss_for_grad)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, 5)
    rng_list = list(rngs)

    # Accumulate the gradients N at a time
    N = 2

    state_encoder_grads = state_encoder_grad_fn(
        state_encoder_state.params, rng_list.pop()
    )
    action_encoder_grads = action_encoder_grad_fn(
        action_encoder_state.params, rng_list.pop()
    )
    transition_model_grads = transition_model_grad_fn(
        transition_model_state.params, rng_list.pop()
    )
    state_decoder_grads = state_decoder_grad_fn(
        state_decoder_state.params, rng_list.pop()
    )
    action_decoder_grads = action_decoder_grad_fn(
        action_decoder_state.params, rng_list.pop()
    )

    return (
        state_encoder_grads,
        action_encoder_grads,
        transition_model_grads,
        state_decoder_grads,
        action_decoder_grads,
    )


def train_step(
    key,
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    transition_model_state,
    rollout_result,
    action_bounds,
    dt,
):
    """Train for a single step."""

    rng, key = jax.random.split(key)

    def tree_zeros_like(tree):
        return jax.tree_map(lambda x: jnp.zeros_like(x), tree)

    def tree_sum(t1, t2):
        return jax.tree_map(lambda x, y: x + y, t1, t2)

    # Get grads in batches
    batch_size = 2
    state_encoder_cum_grad = tree_zeros_like(state_encoder_state.params)
    action_encoder_cum_grad = tree_zeros_like(action_encoder_state.params)
    transition_model_cum_grad = tree_zeros_like(transition_model_state.params)
    state_decoder_cum_grad = tree_zeros_like(state_decoder_state.params)
    action_decoder_cum_grad = tree_zeros_like(action_decoder_state.params)

    for i in range(batch_size):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, rollout_result[0].shape[0])
        (
            this_batch_state_encoder_grad,
            this_batch_action_encoder_grad,
            this_batch_transition_model_grad,
            this_batch_state_decoder_grad,
            this_batch_action_decoder_grad,
        ) = get_grads(
            rng,
            state_encoder_state,
            action_encoder_state,
            state_decoder_state,
            action_decoder_state,
            transition_model_state,
            (
                rollout_result[0][start_index:end_index],
                rollout_result[1][start_index:end_index],
            ),
            action_bounds,
            dt,
        )

        state_encoder_cum_grad = tree_sum(
            this_batch_state_encoder_grad, state_encoder_cum_grad
        )
        action_encoder_cum_grad = tree_sum(
            this_batch_action_encoder_grad, action_encoder_cum_grad
        )
        transition_model_cum_grad = tree_sum(
            this_batch_transition_model_grad, transition_model_cum_grad
        )
        state_decoder_cum_grad = tree_sum(
            this_batch_state_decoder_grad, state_decoder_cum_grad
        )
        action_decoder_cum_grad = tree_sum(
            this_batch_action_decoder_grad, action_decoder_cum_grad
        )

    # def process_grad(grad):
    #     # Clip norm of grads to be 1.0
    #     norm = jnp.linalg.norm(grad)
    #     ratio = jnp.minimum(1.0, 1.0 / norm)
    #     grad = grad * ratio

    #     # nan to zero
    #     grad = jnp.nan_to_num(grad)

    #     return grad

    # state_encoder_grads = jax.tree_map(process_grad, state_encoder_grads)
    # action_encoder_grads = jax.tree_map(process_grad, action_encoder_grads)
    # transition_model_grads = jax.tree_map(process_grad, transition_model_grads)
    # state_decoder_grads = jax.tree_map(process_grad, state_decoder_grads)
    # action_decoder_grads = jax.tree_map(process_grad, action_decoder_grads)

    state_encoder_state = state_encoder_state.apply_gradients(
        grads=state_encoder_cum_grad
    )
    action_encoder_state = action_encoder_state.apply_gradients(
        grads=action_encoder_cum_grad
    )
    transition_model_state = transition_model_state.apply_gradients(
        grads=transition_model_cum_grad
    )
    state_decoder_state = state_decoder_state.apply_gradients(
        grads=state_decoder_cum_grad
    )
    action_decoder_state = action_decoder_state.apply_gradients(
        grads=action_decoder_cum_grad
    )

    return (
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
    )


def compute_metrics(
    key,
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    transition_model_state,
    rollout_result,
    action_bounds,
    dt,
):
    rng, key = jax.random.split(key)
    state_encoder_loss_val = state_encoder_loss(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        action_bounds,
        dt,
        state_encoder_state.params,
    )
    rng, key = jax.random.split(key)
    action_encoder_loss_val = action_encoder_loss(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        action_bounds,
        dt,
        action_encoder_state.params,
    )
    rng, key = jax.random.split(key)
    transition_model_loss_val = transition_model_loss(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        dt,
        transition_model_state.params,
    )

    rng, key = jax.random.split(key)
    state_decoder_loss_val = state_decoder_loss(
        rng,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        state_decoder_state.params,
    )
    action_decoder_loss_val = action_decoder_loss(
        rng,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        action_decoder_state.params,
    )

    # metric_updates = state_encoder_state.metrics.single_from_model_output(
    #     loss=state_encoder_loss_val
    # )
    # metrics = state_encoder_state.metrics.merge(metric_updates)
    # state_encoder_state = state_encoder_state.replace(metrics=metrics)

    # metric_updates = action_encoder_state.metrics.single_from_model_output(
    #     loss=action_encoder_loss_val
    # )
    # metrics = action_encoder_state.metrics.merge(metric_updates)
    # action_encoder_state = action_encoder_state.replace(metrics=metrics)

    # metric_updates = transition_model_state.metrics.single_from_model_output(
    #     loss=transition_model_loss_val
    # )
    # metrics = transition_model_state.metrics.merge(metric_updates)
    # transition_model_state = transition_model_state.replace(metrics=metrics)

    # metric_updates = state_decoder_state.metrics.single_from_model_output(
    #     loss=state_decoder_loss_val
    # )
    # metrics = state_decoder_state.metrics.merge(metric_updates)
    # state_decoder_state = state_decoder_state.replace(metrics=metrics)

    # metric_updates = action_decoder_state.metrics.single_from_model_output(
    #     loss=action_decoder_loss_val
    # )
    # metrics = action_decoder_state.metrics.merge(metric_updates)
    # action_decoder_state = action_decoder_state.replace(metrics=metrics)

    msg = (
        "Losses:\n"
        + "\tState Encoder: {}\n"
        + "\tAction Encoder: {}\n"
        + "\tTransition Model: {}\n"
        + "\tState Decoder: {}\n"
        + "\tAction Decoder: {}".format(
            state_encoder_loss_val,
            action_encoder_loss_val,
            transition_model_loss_val,
            state_decoder_loss_val,
            action_decoder_loss_val,
        )
    )

    return (
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
        msg,
    )
