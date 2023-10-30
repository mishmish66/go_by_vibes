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
    carry,
    vibe_state,
    vibe_config,
):
    """This policy just samples random actions from the action space."""

    rng, key = jax.random.split(key)
    return random_action(rng, vibe_config.env_config.action_bounds), None


def random_repeat_policy(
    key,
    start_state,
    action_i,
    carry,
    vibe_state,
    vibe_config,
    repeat_prob=0.99,
):
    """This policy just samples random actions from the action space."""

    rng, key = jax.random.split(key)
    rand = jax.random.uniform(rng)

    repeat = rand < repeat_prob

    next_action = carry * repeat + random_action(
        rng, vibe_config.env_config.action_bounds
    ) * (1 - repeat)
    
    return next_action, next_action


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

    def __call__(self, key, state, i, carry, vibe_state, vibe_config):
        # encode the state
        rng, key = jax.random.split(key)
        latent_state = encode_state(rng, state, vibe_state, vibe_config)

        # decode the action
        latent_action = self.actions[i]
        rng, key = jax.random.split(key)
        action = decode_action(
            rng, latent_action, latent_state, vibe_state, vibe_config
        )
        return action, carry


def make_random_traj(
    key,
    start_state,
    vibe_state,
    vibe_config,
    env_cls,
):
    rng, key = jax.random.split(key)
    random_states, random_actions = collect_rollout(
        start_state,
        random_policy,
        None,
        env_cls,
        vibe_state,
        vibe_config,
        rng,
    )

    return random_states, random_actions


# def make_optimized_actions(
def optimize_actions(
    key,
    start_state,
    initial_guess,
    cost_func,
    vibe_state,
    vibe_config: TrainConfig,
    env_cls,
    start_state_idx=0,
    big_step_size=0.5,
    big_steps=512,
    small_step_size=0.005,
    small_steps=512,
):
    horizon = vibe_config.rollout_length

    causal_mask = make_mask(horizon, start_state_idx)

    rng, key = jax.random.split(key)
    latent_start_state = encode_state(rng, start_state, vibe_state, vibe_config)

    def big_scanf(current_plan, key):
        rng, key = jax.random.split(key)
        cost, act_grad = jax.value_and_grad(cost_func)(
            current_plan,
            latent_start_state,
            vibe_state,
            vibe_config,
            rng,
        )

        act_grad_future = einsum(act_grad, causal_mask, "i ..., i -> i ...")

        column_norms = jnp.linalg.norm(act_grad_future, ord=1, axis=-1)
        max_column_idx = jnp.argmax(column_norms)
        column_grad = act_grad_future[max_column_idx]
        column_norm = column_norms[max_column_idx]
        normalized_column_grad = column_grad / column_norm

        old_column = current_plan[max_column_idx]
        new_column = old_column - big_step_size * normalized_column_grad

        next_plan = current_plan.at[max_column_idx].set(new_column)
        return next_plan, (cost, max_column_idx)

    def small_scanf(current_plan, key):
        rng, key = jax.random.split(key)
        cost, act_grad = jax.value_and_grad(cost_func)(
            current_plan,
            latent_start_state,
            vibe_state,
            vibe_config,
            rng,
        )

        act_grad_future = einsum(act_grad, causal_mask, "i ..., i -> i ...")

        next_plan = current_plan - small_step_size * act_grad_future
        return next_plan, cost

    rng, key = jax.random.split(key)
    scan_rng = jax.random.split(rng, big_steps)
    coarse_latent_action_sequence, (big_costs, big_active_inds) = jax.lax.scan(
        big_scanf, initial_guess, scan_rng
    )

    rng, key = jax.random.split(key)
    scan_rng = jax.random.split(rng, small_steps)
    fine_latent_action_sequence, small_costs = jax.lax.scan(
        small_scanf, coarse_latent_action_sequence, scan_rng
    )

    costs = jnp.concatenate([big_costs, small_costs], axis=0)

    return fine_latent_action_sequence, (costs, big_active_inds)


def make_optimize_actor(
    key,
    start_state,
    cost_func,
    vibe_state,
    vibe_config: TrainConfig,
    env_cls,
    big_step_size=0.5,
    big_steps=512,
    small_step_size=0.005,
    small_steps=512,
):
    rng, key = jax.random.split(key)
    random_traj_states, random_traj_actions = make_random_traj(
        rng,
        start_state,
        vibe_state,
        vibe_config,
        env_cls,
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, random_traj_states.shape[0])
    random_latent_states = jax.vmap(encode_state, (0, 0, None, None))(
        rngs, random_traj_states, vibe_state, vibe_config
    )

    rng, key = jax.random.split(key)
    rngs = jax.random.split(rng, random_traj_states.shape[0])
    random_latent_actions = jax.vmap(encode_action, (0, 0, 0, None, None))(
        rngs, random_traj_actions, random_latent_states, vibe_state, vibe_config
    )

    rng, key = jax.random.split(key)
    optimized_actions, (costs, big_active_inds) = optimize_actions(
        rng,
        start_state,
        random_latent_actions,
        cost_func,
        vibe_state,
        vibe_config,
        env_cls,
        0,
        big_step_size,
        big_steps,
        small_step_size,
        small_steps,
    )

    def optimizer_actor(
        key,
        state,
        i,
        carry,
        vibe_state,
        vibe_config,
    ):
        last_guess = carry

        rng, key = jax.random.split(key)
        next_guess = optimize_actions(
            rng,
            state,
            last_guess,
            cost_func,
            vibe_state,
            vibe_config,
            env_cls,
            start_state_idx=i,
            big_steps=16,
            small_steps=48,
        )[0]

        latent_action = next_guess[i]
        rng, key = jax.random.split(key)
        latent_state = encode_state(rng, state, vibe_state, vibe_config)

        rng, key = jax.random.split(key)
        action = decode_action(
            rng, latent_action, latent_state, vibe_state, vibe_config
        )

        return action, next_guess

    return optimizer_actor, optimized_actions, (costs, big_active_inds)


def make_target_conf_policy(
    key,
    start_state,
    vibe_state,
    vibe_config: TrainConfig,
    env_cls,
    target_uncertainty=1e-7,
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

    actor, init_carry, _ = make_optimize_actor(
        key,
        start_state,
        cost_func,
        vibe_state,
        vibe_config,
        env_cls,
    )

    return actor, init_carry


def make_piecewise_actor(a, b, first_b_idx):
    def actor(key, state, i, carry, vibe_state, vibe_config):
        a_carry, b_carry = carry

        def a_case(a_carry, b_carry):
            result, a_carry = a(key, state, i, a_carry, vibe_state, vibe_config)
            return result, (a_carry, b_carry)

        def b_case(a_carry, b_carry):
            result, b_carry = b(key, state, i, b_carry, vibe_state, vibe_config)
            return result, (a_carry, b_carry)
        
        a_or_b = i < first_b_idx
        
        a_result, new_a_carry = a(key, state, i, a_carry, vibe_state, vibe_config)
        b_result, new_b_carry = b(key, state, i, b_carry, vibe_state, vibe_config)
        
        result = a_or_b * a_result + (1 - a_or_b) * b_result
        
        return result, (a_carry, b_carry)

    return actor
