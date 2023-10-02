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

import os


def batchify_loss(loss_fn, args, dict_args, batch_args={}, batchsize=4):
    args_to_be_batched = [
        arg for arg, is_batch_arg in zip(args, batch_args) if is_batch_arg
    ]
    total_size = args_to_be_batched[0].shape[0]
    cumulative_loss = 0.0

    i = 0
    while i < total_size:
        start_index = i
        end_index = min(i + batchsize, total_size)

        args_for_batch = [
            arg[start_index:end_index] if is_batch_arg else arg
            for arg, is_batch_arg in zip(args, batch_args)
        ]
        cumulative_loss += loss_fn(*args_for_batch, **dict_args)

        i += batchsize

    return cumulative_loss


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

    states, actions = rollout_result

    forward_loss = loss_weights[0] * loss_forward(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        dt,
        state_encoder_params=state_encoder_params,
    )

    rng, key = jax.random.split(key)
    # reconstruction_loss = loss_weights[1] * batchify_loss(
    #     loss_reconstruction,
    #     (
    #         rng,
    #         state_encoder_state,
    #         action_encoder_state,
    #         state_decoder_state,
    #         action_decoder_state,
    #         states,
    #         actions,
    #     ),
    #     {"state_encoder_params": state_encoder_params},
    #     (True, False, False, False, False, True, True),
    # )
    reconstruction_loss = loss_weights[1] * loss_reconstruction(
        rng,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        states,
        actions,
        state_encoder_params=state_encoder_params,
    )

    rng, key = jax.random.split(key)
    smoothness_loss = loss_weights[2] * loss_smoothness(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        dt,
        state_encoder_params=state_encoder_params,
    )

    rng, key = jax.random.split(key)
    dispersion_loss = loss_weights[3] * loss_disperse(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        action_bounds,
        dt,
        state_encoder_params=state_encoder_params,
    )

    # jax.debug.print(
    #     "State Encoder Losses:\n"
    #     + "\tForward: {}\n"
    #     + "\tReconstruction: {}\n"
    #     + "\tSmoothness: {}\n"
    #     + "\tDispersion: {}",
    #     forward_loss,
    #     reconstruction_loss,
    #     smoothness_loss,
    #     dispersion_loss,
    # )

    return forward_loss + reconstruction_loss + smoothness_loss + dispersion_loss, {
        "forward": forward_loss,
        "reconstruction": reconstruction_loss,
        "smoothness": smoothness_loss,
        "dispersion": dispersion_loss,
    }


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

    states, actions = rollout_result

    forward_loss = loss_forward(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
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
        states,
        actions,
        action_encoder_params=action_encoder_params,
    )

    rng, key = jax.random.split(key)
    smoothness_loss = loss_smoothness(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        dt,
        action_encoder_params=action_encoder_params,
    )

    rng, key = jax.random.split(key)
    dispersion_loss = loss_disperse(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
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

    # jax.debug.print(
    #     "Action Encoder Losses:\n"
    #     + "\tForward: {}\n"
    #     + "\tReconstruction: {}\n"
    #     + "\tSmoothness: {}\n"
    #     + "\tDispersion: {}\n"
    #     + "\tCondensation: {}",
    #     forward_loss,
    #     reconstruction_loss,
    #     smoothness_loss,
    #     dispersion_loss,
    #     condensation_loss,
    # )

    return forward_loss + reconstruction_loss + smoothness_loss + condensation_loss, {
        "forward": forward_loss,
        "reconstruction": reconstruction_loss,
        "smoothness": smoothness_loss,
        "dispersion": dispersion_loss,
        "condensation": condensation_loss,
    }


def transition_model_loss(
    key,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    rollout_result,
    dt,
    transition_model_params,
):
    states, actions = rollout_result

    rng, key = jax.random.split(key)
    forward_loss = loss_forward(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        dt,
        transition_params=transition_model_params,
    )

    return forward_loss, {"forward": forward_loss}


def state_decoder_loss(
    key,
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    state_decoder_params,
):
    states, actions = rollout_result

    rng, key = jax.random.split(key)
    reconstruction_loss = loss_reconstruction(
        rng,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        states,
        actions,
        state_decoder_params=state_decoder_params,
    )

    return reconstruction_loss, {"reconstruction": reconstruction_loss}


def action_decoder_loss(
    key,
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    action_decoder_params,
):
    states, actions = rollout_result

    rng, key = jax.random.split(key)
    reconstruction_loss = loss_reconstruction(
        rng,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        states,
        actions,
        action_decoder_params=action_decoder_params,
    )

    return reconstruction_loss, {"reconstruction": reconstruction_loss}


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
        )[0]

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
        )[0]

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
        )[0]

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
        )[0]

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
        )[0]

    def state_encoder_loss_for_info(state_encoder_params, key):
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
        )[1]

    def action_encoder_loss_for_info(action_encoder_params, key):
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
        )[1]

    def transition_model_loss_for_info(transition_model_params, key):
        rng, key = jax.random.split(key)
        return transition_model_loss(
            rng,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            rollout_result,
            dt,
            transition_model_params,
        )[1]

    def state_decoder_loss_for_info(state_decoder_params, key):
        rng, key = jax.random.split(key)
        return state_decoder_loss(
            rng,
            state_encoder_state,
            action_encoder_state,
            state_decoder_state,
            action_decoder_state,
            rollout_result,
            state_decoder_params,
        )[1]

    def action_decoder_loss_for_info(action_decoder_params, key):
        rng, key = jax.random.split(key)
        return action_decoder_loss(
            rng,
            state_encoder_state,
            action_encoder_state,
            state_decoder_state,
            action_decoder_state,
            rollout_result,
            action_decoder_params,
        )[1]

    action_encoder_grad_fn = jax.grad(action_encoder_loss_for_grad)
    state_encoder_grad_fn = jax.grad(state_encoder_loss_for_grad)
    transition_model_grad_fn = jax.grad(transition_model_loss_for_grad)
    state_decoder_grad_fn = jax.grad(state_decoder_loss_for_grad)
    action_decoder_grad_fn = jax.grad(action_decoder_loss_for_grad)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, 5)
    rng_list = list(rngs)

    state_encoder_grads = jax.jit(state_encoder_grad_fn)(
        state_encoder_state.params, rng_list.pop()
    )
    action_encoder_grads = jax.jit(action_encoder_grad_fn)(
        action_encoder_state.params, rng_list.pop()
    )
    transition_model_grads = jax.jit(transition_model_grad_fn)(
        transition_model_state.params, rng_list.pop()
    )
    state_decoder_grads = jax.jit(state_decoder_grad_fn)(
        state_decoder_state.params, rng_list.pop()
    )
    action_decoder_grads = jax.jit(action_decoder_grad_fn)(
        action_decoder_state.params, rng_list.pop()
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, 5)
    rng_list = list(rngs)

    state_encoder_loss_info = jax.jit(state_encoder_loss_for_info)(
        state_encoder_state.params, rng_list.pop()
    )
    action_encoder_loss_info = jax.jit(action_encoder_loss_for_info)(
        action_encoder_state.params, rng_list.pop()
    )
    transition_model_loss_info = jax.jit(transition_model_loss_for_info)(
        transition_model_state.params, rng_list.pop()
    )
    state_decoder_loss_info = jax.jit(state_decoder_loss_for_info)(
        state_decoder_state.params, rng_list.pop()
    )
    action_decoder_loss_info = jax.jit(action_decoder_loss_for_info)(
        action_decoder_state.params, rng_list.pop()
    )

    return (
        state_encoder_grads,
        action_encoder_grads,
        transition_model_grads,
        state_decoder_grads,
        action_decoder_grads,
        {
            "state_encoder_losses": state_encoder_loss_info,
            "action_encoder_losses": action_encoder_loss_info,
            "transition_model_losses": transition_model_loss_info,
            "state_decoder_losses": state_decoder_loss_info,
            "action_decoder_losses": action_decoder_loss_info,
        },
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

    (
        state_encoder_grads,
        action_encoder_grads,
        transition_model_grads,
        state_decoder_grads,
        action_decoder_grads,
        loss_infos,
    ) = get_grads(
        rng,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        transition_model_state,
        rollout_result,
        action_bounds,
        dt,
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
        loss_infos,
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
    state_encoder_loss_val, _ = state_encoder_loss(
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
    action_encoder_loss_val, _ = action_encoder_loss(
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
    transition_model_loss_val, _ = transition_model_loss(
        rng,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        rollout_result,
        dt,
        transition_model_state.params,
    )

    rng, key = jax.random.split(key)
    state_decoder_loss_val, _ = state_decoder_loss(
        rng,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        state_decoder_state.params,
    )
    action_decoder_loss_val, _ = action_decoder_loss(
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
        + f"\tState Encoder: {state_encoder_loss_val}\n"
        + f"\tAction Encoder: {action_encoder_loss_val}\n"
        + f"\tTransition Model: {transition_model_loss_val}\n"
        + f"\tState Decoder: {state_decoder_loss_val}\n"
        + f"\tAction Decoder: {action_decoder_loss_val}"
    )

    return (
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
        msg,
    )


def make_info_msgs(infos):
    paths_and_strings = {
        os.path.join(outer, f"{inner}.txt"): info
        for outer, inner_info in infos.items()
        for inner, info in inner_info.items()
    }

    return paths_and_strings


def merge_info_msgs(paths_and_stringses):
    return {
        path: sum([paths_and_strings[path] for paths_and_strings in paths_and_stringses])
        for path in paths_and_stringses[0].keys()
    }


def dump_infos(location, infos, epoch, start_i, end_i):
    paths_and_strings = {
        os.path.join(outer, f"{inner}.txt"): info
        for outer, inner_info in infos.items()
        for inner, info in inner_info.items()
    }

    for path, string in paths_and_strings.items():
        filepath = os.path.join(location, path)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a") as f:
            string_to_add = f"{string}\t\tEpoch {epoch}, Samples {start_i} - {end_i}\n"
            f.write(string_to_add)
