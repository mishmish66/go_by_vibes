import jax

import jax.numpy as jnp

from .gen.mass_matrix import mass_matrix
from .gen.bias_forces import bias_forces

from physics.positions import make_positions

from physics.contact_solver import iterative_solver

import timeit

from physics.visualize import animate

from einops import einops, einsum

import matplotlib.pyplot as plt


def penetration(positions, position_grads=None):
    return jnp.array(
        [-2 - positions.lfoot_pos[1], -2 - positions.rfoot_pos[1]], dtype=jnp.float32
    )


def contact_normals(positions, position_grads=None):
    return jnp.array([[0.0, 1.0], [0.0, 1.0]], dtype=jnp.float32)


def tangent_amount(positions, position_grads=None):
    return jnp.array(
        [positions.lfoot_pos[0], positions.rfoot_pos[0]],
        dtype=jnp.float32,
    )


def contact_normal_grads(positions, position_grads):
    penetration_jac = jax.jacfwd(penetration, 0)(positions)

    flat_penetration_jac = jax.tree_util.tree_flatten(penetration_jac)[0]
    flat_position_grads = jax.tree_util.tree_flatten(position_grads)[0]

    results = []
    for penetration_jac_pos, position_grads_pos in zip(
        flat_penetration_jac, flat_position_grads
    ):
        pos_result = einsum(penetration_jac_pos, position_grads_pos, "C e, e J -> C J")
        results.append(pos_result)

    result_stack = jnp.stack(results, axis=0)

    return jnp.sum(result_stack, axis=0)


def contact_tangent_grads(positions, position_grads):
    tangent_amount_jac = jax.jacfwd(tangent_amount, 0)(positions)

    flat_tangent_amount_jac = jax.tree_util.tree_flatten(tangent_amount_jac)[0]
    flat_position_grads = jax.tree_util.tree_flatten(position_grads)[0]

    results = []

    for tangent_amount_jac_pos, position_grad_pos in zip(
        flat_tangent_amount_jac, flat_position_grads
    ):
        pos_result = einsum(
            tangent_amount_jac_pos, position_grad_pos, "C e, e J -> C J"
        )
        results.append(pos_result)

    result_stack = jnp.stack(results, axis=0)

    return jnp.sum(result_stack, axis=0)


def step(
    q,
    qd,
    mass_config,
    shape_config,
    control=None,
    dt=0.01,
):
    positions = make_positions(q, shape_config)
    position_grads = jax.jacfwd(make_positions, 0)(q, shape_config)

    penetration_now = penetration(positions)
    contact_normals_now = contact_normals(positions)
    contact_tangents_now = tangent_amount(positions)
    contact_normal_grads_now = contact_normal_grads(positions, position_grads)
    contact_tangent_grads_now = contact_tangent_grads(positions, position_grads)
    
    # Force the largest magnitude value in contact_normal_grads_now to be at least 0.01
    # def normalize_grad(grad):
    #     mags = jnp.abs(grad)
    #     max_mag = jnp.max(mags)
    #     factor = jnp.maximum(max_mag, 0.01) / max_mag
        
    #     return grad * factor
    
    # jax.vmap(normalize_grad)(contact_normal_grads_now)

    bias_force_now = bias_forces(q, qd, mass_config, shape_config, 9.81)
    # add a joint damper to the bias force
    bias_force_now = bias_force_now + jnp.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]) * qd * 0.1
    # Set control to 0 if it is none
    control = jax.lax.cond(
        control is None,
        lambda: jnp.zeros_like(bias_force_now),
        lambda: control,
    )

    tau = bias_force_now + control

    qdd_pre_contact = jnp.linalg.solve(
        mass_matrix(
            q,
            qd,
            mass_config,
            shape_config,
        ),
        tau,
    )
    qd_pre_contact = qd + qdd_pre_contact * dt

    contact_force_now = iterative_solver(
        q,
        qd_pre_contact,
        contact_normals_now,
        contact_tangents_now,
        penetration_now,
        contact_normal_grads_now,
        contact_tangent_grads_now,
        mass_config,
        shape_config,
        coeff_rest=0.0001,
        coeff_fric=0.7,
        dt=dt,
    )
    
    # Apply a non physical correction to push the contact points out of the ground
    non_physical_correction = einsum(contact_normal_grads_now, penetration_now, "c d, c -> d") * dt * 2.5
    q = q - non_physical_correction

    qdd_contact = jnp.linalg.solve(
        mass_matrix(
            q,
            qd_pre_contact,
            mass_config,
            shape_config,
        ),
        contact_force_now,
    )

    qd = qd_pre_contact + qdd_contact * dt
    q = q + qd * dt

    return (q, qd)
