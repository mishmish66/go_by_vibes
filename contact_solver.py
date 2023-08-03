from mass_matrix import mass_matrix

from foot_positions import lfoot_pos, rfoot_pos

from einops import einsum, reduce

import jax
import jax.numpy as jnp


def ddt(f, q, q_dot, q_ddot):
    df_dq = jax.jacfwd(f, 0)
    df_dq_dot = jax.jacfwd(f, 1)

    df_dt = df_dq(q, q_dot) @ q_dot + df_dq_dot(q, q_dot) @ q_ddot
    return df_dt


def iterative_solver(
    q,
    q_dot,
    contact_normals,
    contact_tangents,
    contact_penetrations,
    contact_normal_grads,
    contact_tangent_grads,
    mass_config,
    shape_config,
    coeff_rest,  # Coefficient of restitution for each contact
    coeff_fric,  # Coefficient of friction for each contact
    dt,
):
    joint_space_mass = mass_matrix(q, q_dot, mass_config, shape_config)
    joint_space_mass_inv = jnp.linalg.pinv(joint_space_mass)

    contact_normal_masses = 1 / einsum(
        contact_normal_grads,
        joint_space_mass_inv,
        contact_normal_grads,
        "C i, i j, C j -> C",
    )

    contact_tangent_masses = 1 / einsum(
        contact_tangent_grads,
        joint_space_mass_inv,
        contact_tangent_grads,
        "C i, i j, C j -> C",
    )

    def contact_normal_speeds_func(
        q_dot_int,
        contact_normal_grads=contact_normal_grads,
    ):
        return einsum(
            contact_normal_grads,
            q_dot_int,
            "C i, i -> C",
        )

    def contact_tangent_speeds_func(
        q_dot_int,
        contact_tangent_grads=contact_tangent_grads,
    ):
        return einsum(
            contact_tangent_grads,
            q_dot,
            "C i, i -> C",
        )

    contact_normal_speeds_init = contact_normal_speeds_func(q_dot)

    active_contacts = jnp.where(contact_penetrations > 0, 1, 0)
    active_contacts = jnp.where(contact_normal_speeds_init < 0, 0, active_contacts)
    post_contact_speed = -1 * contact_normal_speeds_init * coeff_rest

    F_c_joint_init = jnp.zeros_like(q)
    q_dot_init = q_dot

    max_it = 20

    def compute_residuals(F_c_joint_int):
        qd_int = q_dot + (joint_space_mass_inv @ F_c_joint_int) * dt
        contact_normal_speeds_int = contact_normal_speeds_func(qd_int)
        error_vec = contact_normal_speeds_int - post_contact_speed
        residual_vec = jnp.square(error_vec)

        return residual_vec

    def while_cond(
        while_carry_pack,
        epsilon=1e-6,
    ):
        F_c_joint_int, it = while_carry_pack
        residuals = compute_residuals(F_c_joint_int) * active_contacts

        # print residuals if not nan
        # jax.lax.cond(
        #     jnp.any(jnp.isnan(residuals)),
        #     lambda: None,
        #     lambda: jax.debug.print("residuals: {x}", x=residuals),
        # )

        return jax.lax.cond(
            it < max_it,
            lambda: jnp.any(residuals > epsilon),
            lambda: False,
        )

    def update(while_carry_pack):
        F_c_joint_init, it = while_carry_pack
        # jax.debug.print("it: {x}", x=it)

        indices = jnp.arange(0, jnp.shape(contact_tangents)[0])

        def scanf_normal(for_carry_pack, i):
            F_c_joint_int = for_carry_pack
            # jax.debug.print("F_c_joint_int: {x}", x=F_c_joint_int)
            q_dot_int = q_dot_init + (joint_space_mass_inv @ F_c_joint_int) * dt

            contact_active = active_contacts[i]
            contact_normal = contact_normals[i]
            contact_tangent = contact_tangents[i]
            contact_normal_grad = contact_normal_grads[i]
            contact_tangent_grad = contact_tangent_grads[i]
            contact_normal_mass = contact_normal_masses[i]
            contact_tangent_mass = contact_tangent_masses[i]

            normal_velocity_int = contact_normal_grad.T @ q_dot_int
            speed_error = normal_velocity_int - post_contact_speed[i]
            # jax.debug.print("contact: {i} speed_error: {x}", i=i, x=speed_error)
            acceleration_needed = -speed_error / dt

            delta_contact_f = contact_normal_mass * acceleration_needed
            delta_contact_f = jax.lax.cond(
                contact_active != 0,
                lambda: delta_contact_f,
                lambda: jnp.float32(0),
            )

            delta_joint_f = contact_normal_grad * delta_contact_f

            F_c_joint_int = F_c_joint_int + delta_joint_f

            return F_c_joint_int, delta_contact_f

        post_normal_force, normal_forces = jax.lax.scan(
            scanf_normal, F_c_joint_init, indices
        )

        def scanf_tangent(for_carry_pack, i):
            F_c_joint_int = for_carry_pack
            # jax.debug.print("F_c_joint_int: {x}", x=F_c_joint_int)
            q_dot_int = q_dot_init + (joint_space_mass_inv @ F_c_joint_int) * dt

            contact_active = active_contacts[i]
            contact_normal = contact_normals[i]
            contact_tangent = contact_tangents[i]
            contact_normal_grad = contact_normal_grads[i]
            contact_tangent_grad = contact_tangent_grads[i]
            contact_normal_mass = contact_normal_masses[i]
            contact_tangent_mass = contact_tangent_masses[i]
            normal_force = normal_forces[i]

            tangent_velocity_int = contact_tangent_grad.T @ q_dot_int
            speed_error = tangent_velocity_int
            # jax.debug.print("contact: {i} speed_error: {x}", i=i, x=speed_error)
            acceleration_needed = -speed_error / dt

            delta_contact_f = contact_tangent_mass * acceleration_needed

            delta_contact_f = jax.lax.cond(
                contact_active != 0,
                lambda: delta_contact_f,
                lambda: jnp.float32(0),
            )

            delta_contact_f = jnp.clip(
                delta_contact_f,
                -coeff_fric * normal_force,
                coeff_fric * normal_force,
            )

            delta_joint_f = contact_tangent_grad * delta_contact_f

            F_c_joint_int = F_c_joint_int + delta_joint_f

            return F_c_joint_int, delta_contact_f

        post_tangent_force, friction_force = jax.lax.scan(
            scanf_tangent,
            post_normal_force,
            indices,
        )
        final_force = post_tangent_force
        # jax.debug.print("final_force: {x}", x=final_force)
        # Here put the friction loop
        return final_force, it + 1

    solution_force, it = jax.lax.while_loop(while_cond, update, (F_c_joint_init, 0))
    # jax.lax.cond(
    #     it == max_it,
    #     lambda: jax.debug.print("max it reached"),
    #     lambda: None,
    # )
    solution_force = solution_force
    return solution_force


# baumgarte not working
def baumgarte_solver(
    q,
    q_dot,
    contact_normals,
    contact_penetration,
    contact_normal_grads,
    contact_tangent_grads,
    mass_config,
    shape_config,
    # gravity,
    # coeff_rest,  # Coefficient of restitution for each contact
    coeff_fric,  # Coefficient of friction for each contact
    dt,
):
    joint_space_mass = mass_matrix(q, q_dot, mass_config, shape_config)
    joint_space_mass_inv = jnp.linalg.pinv(joint_space_mass)

    contact_normal_masses = 1 / einsum(
        contact_normal_grads,
        joint_space_mass_inv,
        contact_normal_grads,
        "C i, i j, C j -> C",
    )

    contact_tangent_masses = 1 / einsum(
        contact_tangent_grads,
        joint_space_mass_inv,
        contact_tangent_grads,
        "C i, i j, C j -> C",
    )

    contact_normal_speeds = einsum(
        contact_normal_grads,
        q_dot,
        "C i, i -> C",
    )

    contact_tangent_speeds = einsum(
        contact_tangent_grads,
        q_dot,
        "C i, i -> C",
    )

    tangent_forces = -1 * contact_tangent_speeds * contact_tangent_masses / dt

    freq = 1 / dt / 10
    stiffness = 2 * contact_normal_masses * freq**2
    # damping = jnp.sqrt(4 * contact_normal_masses * stiffness)
    damping = jnp.sqrt(4 * contact_normal_masses * stiffness)

    # jax.debug.print("stiff: {x}", x=stiffness)
    # jax.debug.print("damp: {x}", x=damping)

    # stiffness = 100
    # damping = 1

    normal_forces = (
        stiffness * (contact_penetration - 0) - damping * contact_normal_speeds
    )

    tangent_forces = jnp.clip(
        tangent_forces,
        -coeff_fric * normal_forces,
        coeff_fric * normal_forces,
    )

    normal_forces = jnp.where(contact_penetration < 0, 0, normal_forces)
    tangent_forces = jnp.where(contact_penetration < 0, 0, tangent_forces)

    normal_joint_force = contact_normal_grads.T @ normal_forces
    tangent_joint_force = contact_tangent_grads.T @ tangent_forces

    total_joint_force = normal_joint_force + tangent_joint_force

    return total_joint_force
