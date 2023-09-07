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
from nets import encoded_state_dim, encoded_action_dim
from loss import (
    loss_forward,
    loss_reconstruction,
    loss_smoothness,
    loss_state_regularization,
    loss_action_regularization,
    sample_gaussian,
)
from rollout import collect_rollout


def collect_latent_rollout(
    state_z_0,
    mass_config,
    shape_config,
    actions,
    transition_model_state,
    dt,
    key,
    deterministic=False,
):
    """Collect a rollout using the latent transitions model and predefined actions."""

    def scanf(carry, action):
        z, key = carry
        state_action = jnp.concatenate([z, action], axis=-1)
        net_out = transition_model_state.apply_fn(
            {"params": transition_model_state.params}, state_action
        )

        if deterministic:
            z_hat = net_out[:encoded_state_dim]
            return z_hat, z_hat

        rng, key = jax.random.split(key)
        z = sample_gaussian(net_out, rng)

        return (z, key), z

    _, latents = jax.lax.scan(scanf, (state_z_0, key), actions)

    return latents


@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output("loss")


class TrainState(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, dim, rng, learning_rate):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([dim]))[
        "params"
    ]  # initialize parameters by passing a template image
    tx = optax.lion(learning_rate)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx, metrics=Metrics.empty()
    )


def state_encoder_loss(
    state_encoder_params,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    key,
    loss_weights=[1.0, 1.0, 1.0, 1.0],  # [1e6, 1.0, 1e3, 1e-6],
):
    rng, key = jax.random.split(key)

    forward_loss = loss_weights[0] * loss_forward(
        state_encoder_params,
        action_encoder_state.params,
        transition_model_state.params,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        rng,
    )

    reconstruction_loss = loss_weights[1] * loss_reconstruction(
        state_encoder_params,
        action_encoder_state.params,
        state_decoder_state.params,
        action_decoder_state.params,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        key,
    )

    smoothness_loss = loss_weights[2] * loss_smoothness(
        state_encoder_params,
        action_encoder_state.params,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        rng,
    )

    regularization_loss = loss_weights[3] * loss_state_regularization(
        state_encoder_params,
        state_encoder_state,
        action_encoder_state,
        rollout_result,
        rng,
    )

    jax.debug.print(
        "State Encoder Losses:\n"
        + "\tForward: {}\n"
        + "\tReconstruction: {}\n"
        + "\tSmoothness: {}\n"
        + "\tRegularization: {}",
        forward_loss,
        reconstruction_loss,
        smoothness_loss,
        regularization_loss,
    )

    return forward_loss + reconstruction_loss + smoothness_loss + regularization_loss


def action_encoder_loss(
    action_encoder_params,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    key,
):
    rng, key = jax.random.split(key)

    forward_loss = loss_forward(
        state_encoder_state.params,
        action_encoder_params,
        transition_model_state.params,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        rng,
    )

    reconstruction_loss = loss_reconstruction(
        state_encoder_state.params,
        action_encoder_params,
        state_decoder_state.params,
        action_decoder_state.params,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        key,
    )

    smoothness_loss = loss_smoothness(
        state_encoder_state.params,
        action_encoder_params,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        rng,
    )

    regularization_loss = loss_action_regularization(
        action_encoder_params,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        0.1,
        rng,
    )

    jax.debug.print(
        "Action Encoder Losses:\n"
        + "\tForward: {}\n"
        + "\tReconstruction: {}\n"
        + "\tSmoothness: {}\n"
        + "\tRegularization: {}",
        forward_loss,
        reconstruction_loss,
        smoothness_loss,
        regularization_loss,
    )

    return forward_loss + reconstruction_loss + smoothness_loss + regularization_loss


def transition_model_loss(
    transition_model_params,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    key,
):
    rng, key = jax.random.split(key)

    forward_loss = loss_forward(
        state_encoder_state.params,
        action_encoder_state.params,
        transition_model_params,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        rng,
    )

    return forward_loss


def state_decoder_loss(
    state_decoder_params,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    key,
):
    rng, key = jax.random.split(key)

    reconstruction_loss = loss_reconstruction(
        state_encoder_state.params,
        action_encoder_state.params,
        state_decoder_params,
        action_decoder_state.params,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        key,
    )

    return reconstruction_loss


def action_decoder_loss(
    action_decoder_params,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    key,
):
    rng, key = jax.random.split(key)

    reconstruction_loss = loss_reconstruction(
        state_encoder_state.params,
        action_encoder_state.params,
        state_decoder_state.params,
        action_decoder_params,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        key,
    )

    return reconstruction_loss


def train_step(
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    transition_model_state,
    rollout_result,
    key,
):
    """Train for a single step."""

    rng, key = jax.random.split(key)

    def state_encoder_loss_for_grad(state_encoder_params, key):
        return state_encoder_loss(
            state_encoder_params,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
            rollout_result,
            key,
        )

    def action_encoder_loss_for_grad(action_encoder_params, key):
        return action_encoder_loss(
            action_encoder_params,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
            rollout_result,
            key,
        )

    def transition_model_loss_for_grad(transition_model_params, key):
        return transition_model_loss(
            transition_model_params,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
            rollout_result,
            key,
        )

    def state_decoder_loss_for_grad(state_decoder_params, key):
        return state_decoder_loss(
            state_decoder_params,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
            rollout_result,
            key,
        )

    def action_decoder_loss_for_grad(action_decoder_params, key):
        return action_decoder_loss(
            action_decoder_params,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            state_decoder_state,
            action_decoder_state,
            rollout_result,
            key,
        )

    action_encoder_grad_fn = jax.grad(action_encoder_loss_for_grad)
    state_encoder_grad_fn = jax.grad(state_encoder_loss_for_grad)
    transition_model_grad_fn = jax.grad(transition_model_loss_for_grad)
    state_decoder_grad_fn = jax.grad(state_decoder_loss_for_grad)
    action_decoder_grad_fn = jax.grad(action_decoder_loss_for_grad)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(key, 5)
    rng_list = list(rngs)

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

    def process_grad(grad):
        # Clip norm of grads to be 1.0
        norm = jnp.linalg.norm(grad)
        ratio = jnp.minimum(1.0, 1.0 / norm)
        grad = grad * ratio
        
        # nan to zero
        grad = jnp.nan_to_num(grad)
        
        return grad

    state_encoder_grads = jax.tree_map(process_grad, state_encoder_grads)
    action_encoder_grads = jax.tree_map(process_grad, action_encoder_grads)
    transition_model_grads = jax.tree_map(process_grad, transition_model_grads)
    state_decoder_grads = jax.tree_map(process_grad, state_decoder_grads)
    action_decoder_grads = jax.tree_map(process_grad, action_decoder_grads)

    state_encoder_state = state_encoder_state.apply_gradients(grads=state_encoder_grads)
    action_encoder_state = action_encoder_state.apply_gradients(
        grads=action_encoder_grads
    )
    transition_model_state = transition_model_state.apply_gradients(
        grads=transition_model_grads
    )
    state_decoder_state = state_decoder_state.apply_gradients(grads=state_decoder_grads)
    action_decoder_state = action_decoder_state.apply_gradients(
        grads=action_decoder_grads
    )

    return (
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
    )


def compute_metrics(
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    transition_model_state,
    rollout_result,
    key,
):
    state_encoder_loss_val = state_encoder_loss(
        state_encoder_state.params,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        key,
    )
    action_encoder_loss_val = action_encoder_loss(
        action_encoder_state.params,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        key,
    )
    transition_model_loss_val = transition_model_loss(
        transition_model_state.params,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        key,
    )
    state_decoder_loss_val = state_decoder_loss(
        state_decoder_state.params,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        key,
    )
    action_decoder_loss_val = action_decoder_loss(
        action_decoder_state.params,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        key,
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

    # jax.debug.print(
    #     "Losses:\n"
    #     + "\tState Encoder: {}\n"
    #     + "\tAction Encoder: {}\n"
    #     + "\tTransition Model: {}\n"
    #     + "\tState Decoder: {}\n"
    #     + "\tAction Decoder: {}",
    #     state_encoder_loss_val,
    #     action_encoder_loss_val,
    #     transition_model_loss_val,
    #     state_decoder_loss_val,
    #     action_decoder_loss_val,
    # )

    return (
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
    )
