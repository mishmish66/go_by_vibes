import jax
from jax import numpy as jnp

import optax

from einops import einsum, rearrange





def mat_gauss_kl(gaussians):
    return jax.vmap(
        jax.vmap(gauss_kl, in_axes=(0, None)),
        in_axis=(None, 0),
    )(gaussians, gaussians)


def loss_info_states(encoded_states):
    encoded_states = rearrange(encoded_states, "r t d -> (r t) d")

    cross_corr = einsum(encoded_states, encoded_states, "i d, j d -> i j")

    labels = jnp.arange(encoded_states.shape[0])
    loss = jnp.sum(
        optax.softmax_cross_entropy_with_integer_labels(
            logits=jnp.log(cross_corr), labels=labels
        )
    )

    return loss

# def loss_smoothness(
#     state_encoder_params,
#     action_encoder_params,
#     state_encoder_state,
#     action_encoder_state,
#     transition_state,
#     key,
# ):
#     encoded_states = state_encoder_state.apply_fn(
#         {"params": state_encoder_params}, rollout_results.states
#     )
#     encoded_actions = action_encoder_state.apply_fn(
#         {"params": action_encoder_params}, rollout_results.actions
#     )

#     next_state_gaussians = transition_state.apply_fn(
#         {"params": transition_params},
#         encoded_states,
#         encoded_actions,
#     )

#     # Sample a random point from each gaussian
#     eval_states = jax.vmap(jax.random.multivariate_normal, (1, 2))(
#         key,
#         next_state_gaussians[..., :encoded_state_dim],
#         jnp.diag(next_state_gaussians[..., encoded_state_dim:]),
#     )

#     # Get the gradient of the probability mass of the predicted next state w.r.t
#     # the distribution's parameters
#     dp_dg = jax.grad(eval_gaussian, argnums=0)(
#         encoded_states,
#         next_state_gaussians,
#     )

#     # Use the chain rule get the gradient of the probability mass of the predicted
#     # next state with respect to the previous state and action
#     dp_dsa = dp_dg @ rollout_results.next_state_jacobians

#     grad_mag_sq = einsum(dp_dsa, dp_dsa, "... d, ... d -> ...")

#     # Force the magnitude of the gradient to be 1
#     loss = jnp.square(grad_mag_sq - 1)

#     return loss