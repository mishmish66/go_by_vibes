import jax

import jax.numpy as jnp

from mass_matrix import mass_matrix
from bias_forces import bias_forces

from positions import make_positions

from contact_solver import iterative_solver

import timeit

from visualize import animate

from einops import einops, einsum

import matplotlib.pyplot as plt

mass_config = jnp.array([1.0, 0.25, 0.25, 0.04, 0.01, 0.01])
shape_config = jnp.array([1.0, 0.25, 0.25])


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


def physics_step(q, qd, control=None, dt=0.01):
    positions = make_positions(q, shape_config)
    position_grads = jax.jacfwd(make_positions, 0)(q, shape_config)

    penetration_now = penetration(positions)
    contact_normals_now = contact_normals(positions)
    contact_tangents_now = tangent_amount(positions)
    contact_normal_grads_now = contact_normal_grads(positions, position_grads)
    contact_tangent_grads_now = contact_tangent_grads(positions, position_grads)

    bias_force_now = bias_forces(q, qd, mass_config, shape_config, 9.81)
    jax.lax.cond(
        control is None,
        lambda q, qd: jnp.zeros_like(bias_force_now),
        control,
        *[q, qd],
    )
    control_force_now = control(q, qd)

    # jax.debug.print("joint_errors: {x}", x=target - q2)
    tau = bias_force_now + control_force_now

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
