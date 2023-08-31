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
from loss import loss_forward, loss_reconstruction, loss_smoothness, sample_gaussian
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


def train_step(
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    transition_model_state,
    q_0,
    qd_0,
    mass_config,
    shape_config,
    policy,
    dt,
    substep,
    rollout_steps,
    key,
    envs,
):
    """Train for a single step."""
    
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, envs)

    rollout_result = jax.vmap(collect_rollout, in_axes=((0, 0) + (None,) * 6 + (0,)))(
        q_0,
        qd_0,
        mass_config,
        shape_config,
        policy,
        dt,
        substep,
        rollout_steps,
        rngs
    )

    rng, key = jax.random.split(key)

    def state_encoder_loss(state_encoder_params, key):
        rng, key = jax.random.split(key)

        forward_loss = loss_forward(
            state_encoder_params,
            action_encoder_state.params,
            transition_model_state.params,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            rollout_result,
            rng,
        )

        reconstruction_loss = loss_reconstruction(
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

        smoothness_loss = loss_smoothness(
            state_encoder_params,
            action_encoder_state.params,
            transition_model_state.params,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            rollout_result,
            rng,
        )

        return forward_loss + reconstruction_loss + smoothness_loss

    def action_encoder_loss(action_encoder_params, key):
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
            transition_model_state.params,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            rollout_result,
            rng,
        )

        return forward_loss + reconstruction_loss + smoothness_loss

    def transition_model_loss(transition_model_params, key):
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

    def state_decoder_loss(state_decoder_params, key):
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

    def action_decoder_loss(action_decoder_params, key):
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

    action_encoder_grad_fn = jax.grad(action_encoder_loss)
    state_encoder_grad_fn = jax.grad(state_encoder_loss)
    transition_model_grad_fn = jax.grad(transition_model_loss)
    state_decoder_grad_fn = jax.grad(state_decoder_loss)
    action_decoder_grad_fn = jax.grad(action_decoder_loss)

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


def compute_metrics(*, state, batch):
    logits = state.apply_fn({"params": state.params}, batch["image"])
    loss = optax.softmax_cross_entropy_with_integer_labels(
        logits=logits, labels=batch["label"]
    ).mean()
    metric_updates = state.metrics.single_from_model_output(
        logits=logits, labels=batch["label"], loss=loss
    )
    metrics = state.metrics.merge(metric_updates)
    state = state.replace(metrics=metrics)
    return state
