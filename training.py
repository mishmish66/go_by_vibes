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


def sample(x, rng):
    len = x.shape[-1] / 2
    x_hat = x[..., :len]
    x_var = x[..., len:]

    return jax.random.normal(rng, x.shape) * jnp.sqrt(x_var) + x_hat


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

    def scanf(z, action):
        net_out = transition_model_state.apply_fn(
            {"params": transition_model_state}, z, action
        )

        if deterministic:
            z_hat = net_out[:encoded_state_dim]
            return z_hat, z_hat

        z = sample(net_out, key)

        return z, z

    _, latents = jax.lax.scan(scanf, state_z_0, actions)

    return latents


def collect_physics_rollout(
    q_0,
    qd_0,
    mass_config,
    shape_config,
    policy,
    dt,
    substep,
    steps,
):
    """Collect a rollout of physics data."""

    ### Initialize Physics
    def scanf(carry, _):
        q, qd = carry
        q, qd = step(q, qd, mass_config, shape_config, policy, dt / substep)
        return (q, qd), q

    _, qs_sub = jax.lax.scan(scanf, (q_0, qd_0), None, length=steps * substep)

    return qs_sub[::substep]


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


def gauss_kl(p, q):
    dim = p.shape[-1] // 2
    p_var, q_var = p[..., dim:], q[..., dim:]
    p_mean, q_mean = p[..., :dim], q[..., :dim]

    ratio = p_var / q_var
    mahalanobis = jnp.square(q_mean - p_mean) / q_var

    return 0.5 * (jnp.sum(ratio) + jnp.sum(mahalanobis) - dim + jnp.sum(jnp.log(ratio)))


def mat_gauss_kl(gaussians):
    return jax.vmap(
        jax.vmap(gauss_kl, in_axes=(0, None)),
        in_axis=(None, 0),
    )(gaussians, gaussians)


def loss_info_states(encoded_states):
    
    encoded_states = rearrange(encoded_states, "r t d -> (r t) d")
    
    cross_corr = einsum(encoded_states, encoded_states,
                        "i d, j d -> i j")
    
    labels = jnp.arange(encoded_states.shape[0])
    loss = jnp.sum(optax.softmax_cross_entropy_with_integer_labels(
        logits=jnp.log(cross_corr), labels=labels
    ))
    
    return loss


def loss_simulation(encoded_states, next_state_gaussians):
    
    next_state_gaussians = next_state_gaussians[..., :-1, :]
    encoded_states = encoded_states[..., 1:, :]
    
    next_state_gaussians = rearrange(next_state_gaussians, "r t d -> (r t) d")
    encoded_states = rearrange(encoded_states, "r t d -> (r t) d")
    
    def eval_gaussian(gaussian, point):
        dim = gaussian.shape[-1] // 2
        mean = gaussian[..., :dim]
        variance = gaussian[..., dim:]
        return multinorm_pdf(point, mean, jnp.diag(variance))
        
    probs = jax.vmap(eval_gaussian, in_axes=(0, 0, 0))(
        next_state_gaussians, encoded_states
    )

    return -jnp.sum(probs)

def loss_consistency(encoded, recovered):
    


@jax.jit
def train_step(state, batch):
    """Train for a single step."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, batch["image"])
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits, labels=batch["label"]
        ).mean()
        return loss

    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state


@jax.jit
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
