import jax
from jax import numpy as jnp

from training.nets import (
    encoded_state_dim,
    encoded_action_dim,
)

from training.vibe_state import TrainConfig

from einops import einsum

from training.inference import (
    infer_states,
    make_mask,
    get_latent_state_prime_gaussians,
    encode_state,
    encode_action,
    decode_action,
    encoded_state_dim,
    encoded_action_dim,
)

from training.rollout import collect_rollout

# from training.vibe_state import collect_latent_rollout

from training.loss import sample_gaussian

from dataclasses import dataclass


def random_action(key, action_bounds):
    rng, key = jax.random.split(key)
    random_nums = jax.random.uniform(rng, (action_bounds.shape[0],))
    scaled = random_nums * (action_bounds[:, 1] - action_bounds[:, 0])
    scaled_and_shifted = scaled + action_bounds[:, 0]

    return scaled_and_shifted


def random_policy(
    key,
    start_state,
    action_i,
    vibe_state,
    vibe_config,
):
    """This policy just samples random actions from the action space."""

    rng, key = jax.random.split(key)
    return random_action(rng, vibe_config.env_config.action_bounds)


# Down here is not really ready, but I'm just putting it here for now
def make_traj_opt_policy(cost_func):
    def policy(key, start_state, vibe_state, vibe_config):
        pass


@jax.tree_util.register_pytree_node_class
@dataclass
class PresetActor:
    actions: all

    def tree_flatten(self):
        return (self.actions,), None

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children)

    def __call__(self, key, state, i, vibe_state, vibe_config):
        # encode the state
        rng, key = jax.random.split(key)
        latent_state = encode_state(rng, state, vibe_state, vibe_config)

        # decode the action
        latent_action = self.actions[i]
        rng, key = jax.random.split(key)
        action = decode_action(
            rng, latent_action, latent_state, vibe_state, vibe_config
        )
        return action


def make_optimized_actions(
    key,
    start_state,
    cost_func,
    vibe_state,
    vibe_config: TrainConfig,
    env_cls,
    refine_steps=4096,
):
    horizon = vibe_config.rollout_length

    rng, key = jax.random.split(key)
    latent_start_state = encode_state(rng, start_state, vibe_state, vibe_config)

    step_size = 0.05

    def scanf(current_plan, key):
        rng, key = jax.random.split(key)
        cost, act_grad = jax.value_and_grad(cost_func)(
            current_plan,
            latent_start_state,
            vibe_state,
            vibe_config,
            rng,
        )

        next_plan = current_plan - step_size * act_grad
        return next_plan, cost

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, horizon)

    rng, key = jax.random.split(key)
    random_states, random_actions = collect_rollout(
        start_state,
        random_policy,
        env_cls,
        vibe_state,
        vibe_config,
        rng,
    )

    latent_random_states = jax.vmap(encode_state, (0, 0, None, None))(
        rngs, random_states, vibe_state, vibe_config
    )

    latent_random_actions = jax.vmap(encode_action, (0, 0, 0, None, None))(
        rngs, random_actions, latent_random_states, vibe_state, vibe_config
    )

    rng, key = jax.random.split(key)
    scan_rng = jax.random.split(rng, refine_steps)
    encoded_action_sequence, costs = jax.lax.scan(scanf, latent_random_actions, scan_rng)

    return PresetActor(encoded_action_sequence), costs


def make_target_conf_policy(
    key,
    start_state,
    vibe_state,
    vibe_config: TrainConfig,
    env_cls,
    refine_steps=2048,
    target_uncertainty=2.5e-4,
):
    def cost_func(
        latent_actions,
        latent_start_state,
        vibe_state,
        vibe_config,
        key,
    ):
        latent_state_prime_gaussians = get_latent_state_prime_gaussians(
            latent_start_state, latent_actions, vibe_state, vibe_config
        )

        latent_state_prime_gaussian_vars = latent_state_prime_gaussians[
            ..., encoded_state_dim:
        ]
        latent_state_prime_gaussian_var_l1 = jnp.linalg.norm(
            latent_state_prime_gaussian_vars, ord=1, axis=-1
        )

        uncertainty_error = jnp.abs(
            latent_state_prime_gaussian_var_l1 - target_uncertainty
        )

        return jnp.mean(uncertainty_error)

    return make_optimized_actions(
        key,
        start_state,
        cost_func,
        vibe_state,
        vibe_config,
        env_cls,
        refine_steps=refine_steps,
        target_uncertainty=target_uncertainty,
    )[0]


def make_piecewise_actor(a, b, first_b_idx):
    def actor(key, state, i, vibe_state, vibe_config):
        result = jax.lax.cond(
            i < first_b_idx,
            lambda _: a(key, state, i, vibe_state, vibe_config),
            lambda _: b(key, state, i, vibe_state, vibe_config),
            operand=None,
        )
        return result

    return actor
