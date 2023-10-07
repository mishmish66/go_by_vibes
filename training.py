import jax
import jax.numpy as jnp
import jax.lax

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

from other_infos import make_other_infos

import os

import wandb


def multiloss(loss_fn, multiness=16):
    def wrapped_loss_fn(key, *args):
        rngs = jax.random.split(key, multiness)
        loss_runs = jax.vmap(
            loss_fn,
            (0, *((None,) * len(args))),
        )(rngs, *args)
        return jnp.mean(loss_runs)

    return wrapped_loss_fn


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


def create_train_state(module, dims, rng, learning_rate, every_k=1, other={}):
    """Creates an initial `TrainState`."""
    dummy = [jnp.ones(dim) for dim in dims]
    params = module.init(rng, *dummy, **other)[
        "params"
    ]  # initialize parameters by passing a template image

    tx = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.lion(learning_rate),
        ),
        every_k_schedule=every_k,
    )

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
    loss_weights,
):
    trajectory_count = rollout_result[0].shape[0]

    states, actions = rollout_result

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    forward_loss_per_traj = jax.vmap(
        loss_forward, (0, None, None, None, 0, 0, None, None)
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        dt,
        state_encoder_params,
    )
    forward_loss = jnp.mean(forward_loss_per_traj)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    reconstruction_loss_per_traj = jax.vmap(
        multiloss(loss_reconstruction, 16), (0, None, None, None, None, 0, 0, None)
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        states,
        actions,
        state_encoder_params,
    )
    reconstruction_loss = jnp.mean(reconstruction_loss_per_traj)

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    smoothness_loss_per_traj = jax.vmap(
        multiloss(loss_smoothness, 4), (0, None, None, None, 0, 0, None, None)
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        dt,
        state_encoder_params,
    )
    smoothness_loss = jnp.sum(smoothness_loss_per_traj) / trajectory_count

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    dispersion_loss_per_traj = jax.vmap(
        multiloss(loss_disperse, 4), (0, None, None, None, 0, 0, None, None, None)
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        action_bounds,
        dt,
        state_encoder_params,
    )
    dispersion_loss = jnp.sum(dispersion_loss_per_traj) / trajectory_count

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

    shaped_sigmoid_reconstruction_loss = 1 / (1 + jnp.exp(reconstruction_loss + 50))

    gate_value = jax.lax.stop_gradient(shaped_sigmoid_reconstruction_loss)

    gated_forward_loss = forward_loss * gate_value

    return (
        gated_forward_loss * loss_weights["forward"]
        + reconstruction_loss * loss_weights["reconstruction"]
        + smoothness_loss * loss_weights["smoothness"]
        + dispersion_loss * loss_weights["dispersion"],
        (
            {
                "forward": forward_loss_per_traj,
                "reconstruction": reconstruction_loss_per_traj,
                "smoothness": smoothness_loss_per_traj,
                "dispersion": dispersion_loss_per_traj,
                "gate_value": gate_value,
            },
            gate_value,
        ),
    )


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
    weights,
):
    trajectory_count = rollout_result[0].shape[0]

    states, actions = rollout_result

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, states.shape[0])
    forward_loss_per_env = jax.vmap(
        loss_forward, (0, None, None, None, 0, 0, None, None, None)
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        dt,
        None,
        action_encoder_params,
    )
    forward_loss = jnp.sum(forward_loss_per_env) / trajectory_count

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    reconstruction_loss_per_traj = jax.vmap(
        multiloss(loss_reconstruction, 16),
        (0, None, None, None, None, 0, 0, None, None),
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        states,
        actions,
        None,
        action_encoder_params,
    )
    reconstruction_loss = jnp.sum(reconstruction_loss_per_traj) / trajectory_count

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    smoothness_loss_per_traj = jax.vmap(
        multiloss(loss_smoothness, 4), (0, None, None, None, 0, 0, None, None, None)
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        dt,
        None,
        action_encoder_params,
    )
    smoothness_loss = jnp.sum(smoothness_loss_per_traj) / trajectory_count

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    dispersion_loss_per_traj = jax.vmap(
        multiloss(loss_disperse, 4),
        (0, None, None, None, 0, 0, None, None, None, None),
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        action_bounds,
        dt,
        None,
        action_encoder_params,
    )
    dispersion_loss = jnp.sum(dispersion_loss_per_traj) / trajectory_count

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    condensation_loss_per_traj = jax.vmap(
        multiloss(loss_condense, 4), (0, None, None, 0, 0, None, None)
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        states,
        actions,
        action_bounds,
        action_encoder_params,
    )
    condensation_loss = jnp.sum(condensation_loss_per_traj) / trajectory_count

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

    return (
        forward_loss * weights["forward"]
        + reconstruction_loss * weights["reconstruction"]
        + smoothness_loss * weights["smoothness"]
        + dispersion_loss * weights["dispersion"]
        + condensation_loss * weights["condensation"],
        {
            "forward": forward_loss,
            "reconstruction": reconstruction_loss_per_traj,
            "smoothness": smoothness_loss_per_traj,
            "dispersion": dispersion_loss_per_traj,
            "condensation": condensation_loss_per_traj,
        },
    )


def transition_model_loss(
    key,
    state_encoder_state,
    action_encoder_state,
    transition_model_state,
    rollout_result,
    dt,
    transition_model_params,
    weights,
):
    trajectory_count = rollout_result[0].shape[0]

    states, actions = rollout_result

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    forward_loss_per_traj = jax.vmap(
        loss_forward,
        (0, None, None, None, 0, 0, None, None, None, None),
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        states,
        actions,
        dt,
        None,
        None,
        transition_model_params,
    )
    forward_loss = jnp.sum(forward_loss_per_traj) / trajectory_count

    return (
        forward_loss * weights["forward"],
        {"forward": forward_loss_per_traj},
    )


def state_decoder_loss(
    key,
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    state_decoder_params,
    weights,
):
    trajectory_count = rollout_result[0].shape[0]

    states, actions = rollout_result

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    reconstruction_loss = jnp.sum(
        jax.vmap(
            multiloss(loss_reconstruction, 8),
            (0, None, None, None, None, 0, 0, None, None, None),
        )(
            rngs,
            state_encoder_state,
            action_encoder_state,
            state_decoder_state,
            action_decoder_state,
            states,
            actions,
            None,
            None,
            state_decoder_params,
        )
        / trajectory_count
    )

    return (
        reconstruction_loss * weights["reconstruction"],
        {"reconstruction": reconstruction_loss},
    )


def action_decoder_loss(
    key,
    state_encoder_state,
    action_encoder_state,
    state_decoder_state,
    action_decoder_state,
    rollout_result,
    action_decoder_params,
    weights,
):
    trajectory_count = rollout_result[0].shape[0]

    states, actions = rollout_result

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    reconstruction_loss_per_traj = jax.vmap(
        multiloss(loss_reconstruction, 8),
        (0, None, None, None, None, 0, 0, None, None, None, None),
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        states,
        actions,
        None,
        None,
        None,
        action_decoder_params,
    )
    reconstruction_loss = jnp.sum(reconstruction_loss_per_traj) / trajectory_count

    return (
        reconstruction_loss * weights["reconstruction"],
        {"reconstruction": reconstruction_loss_per_traj},
    )


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
    weights,
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
            weights["state_encoder"],
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
            weights["action_encoder"],
        )

    def transition_model_loss_for_grad(transition_model_params, key, gate_value):
        rng, key = jax.random.split(key)
        loss_val, info = transition_model_loss(
            rng,
            state_encoder_state,
            action_encoder_state,
            transition_model_state,
            rollout_result,
            dt,
            transition_model_params,
            weights["transition_model"],
        )

        return loss_val * gate_value, info

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
            weights["state_decoder"],
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
            weights["action_decoder"],
        )

    action_encoder_grad_fn = jax.value_and_grad(
        action_encoder_loss_for_grad, has_aux=True
    )
    state_encoder_grad_fn = jax.value_and_grad(
        state_encoder_loss_for_grad, has_aux=True
    )
    transition_model_grad_fn = jax.value_and_grad(
        transition_model_loss_for_grad, has_aux=True
    )
    state_decoder_grad_fn = jax.value_and_grad(
        state_decoder_loss_for_grad, has_aux=True
    )
    action_decoder_grad_fn = jax.value_and_grad(
        action_decoder_loss_for_grad, has_aux=True
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, 5)
    rng_list = list(rngs)

    (_, (state_encoder_loss_info, gate_value)), state_encoder_grads = jax.jit(
        state_encoder_grad_fn
    )(state_encoder_state.params, rng_list.pop())
    (_, action_encoder_loss_info), action_encoder_grads = jax.jit(
        action_encoder_grad_fn
    )(action_encoder_state.params, rng_list.pop())
    (_, transition_model_loss_info), transition_model_grads = jax.jit(
        transition_model_grad_fn
    )(transition_model_state.params, rng_list.pop(), gate_value)
    (_, state_decoder_loss_info), state_decoder_grads = jax.jit(state_decoder_grad_fn)(
        state_decoder_state.params, rng_list.pop()
    )
    (_, action_decoder_loss_info), action_decoder_grads = jax.jit(
        action_decoder_grad_fn
    )(action_decoder_state.params, rng_list.pop())

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
    weights,
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
        weights,
    )

    def grad_norm(grad):
        jnp.linalg.norm(grad.flatten())

    def grad_max(grad):
        jnp.max(grad.flatten())

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

    def grad_to_list(grad):
        return [e for bk in list(grad.values()) for e in bk.values()]

    def get_grad_norm(grad):
        flat_grads = jax.tree_map(jnp.ravel, grad_to_list(grad))
        return jnp.linalg.norm(jnp.concatenate(flat_grads))

    def get_grad_max(grad):
        flat_grads = jax.tree_map(jnp.ravel, grad_to_list(grad))
        return jnp.max(jnp.abs(jnp.concatenate(flat_grads)))

    state_encoder_grad_norm = get_grad_norm(state_encoder_grads)
    state_encoder_grad_max = get_grad_max(state_encoder_grads)

    action_encoder_grad_norm = get_grad_norm(action_encoder_grads)
    action_encoder_grad_max = get_grad_max(action_encoder_grads)

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

    trajectory_count = rollout_result[0].shape[0]
    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, trajectory_count)
    other_infos = jax.vmap(
        make_other_infos, (0, None, None, None, None, None, 0, 0, None)
    )(
        rngs,
        state_encoder_state,
        action_encoder_state,
        transition_model_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result[0],
        rollout_result[1],
        dt,
    )

    loss_infos = {
        "other_infos": {
            "state_encoder_grad_norm": state_encoder_grad_norm,
            "state_encoder_grad_max": state_encoder_grad_max,
            "action_encoder_grad_norm": action_encoder_grad_norm,
            "action_encoder_grad_max": action_encoder_grad_max,
            **other_infos,
        },
        **loss_infos,
    }

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
    weights,
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
        weights["state_encoder"],
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
        weights["action_encoder"],
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
        weights["transition_model"],
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
        weights["state_decoder"],
    )
    action_decoder_loss_val, _ = action_decoder_loss(
        rng,
        state_encoder_state,
        action_encoder_state,
        state_decoder_state,
        action_decoder_state,
        rollout_result,
        action_decoder_state.params,
        weights["action_decoder"],
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
        path: sum(
            [paths_and_strings[path] for paths_and_strings in paths_and_stringses]
        )
        for path in paths_and_stringses[0].keys()
    }


# def dump_infos(location, infos, rollout):
def dump_infos(path, infos, rollout, every_k):
    # print(f"Dumping infos rollout: {rollout}")
    if not isinstance(infos, dict):
        data = infos
        
        location = os.path.join(*path)
        filepath = f"{location}.txt"

        result = ""
        for epoch_i in range(data.shape[0]):
            for batch_i in range(data.shape[1] // every_k):
                chunk_i = batch_i * every_k
                data_mean = jnp.mean(data[epoch_i, chunk_i, ...])
                num_string = f"{data_mean:.16f}"[:12]
                string_to_add = f"{num_string}\t\tRollout {rollout}, Epoch {epoch_i}, Batch {batch_i}\n"
                result += string_to_add
                
                wandb_log(path, data_mean)

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "a") as f:
            f.write(result)

    else:
        for sub_folder_name, info in infos.items():
            dump_infos([*path, sub_folder_name], info, rollout, every_k)

def dump_to_wandb(infos, chunk_i, every_k):
    if chunk_i % every_k == 0:
        wandb.log(infos)
        
def wandb_log(path, quantity, result_dict = None):
    if len(path) == 0:
        wandb.log(dict)
    else:
        if result_dict is None:
            result_dict = {path[-1]: quantity}
        else:
            result_dict = {path[-1]: result_dict}
        
        wandb_log(path[:-1], None, result_dict)
