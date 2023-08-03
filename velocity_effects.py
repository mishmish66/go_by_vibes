import jax.numpy as jnp

pow = jnp.power
cos = jnp.cos
sin = jnp.sin
sqrt = jnp.sqrt
abs2 = jnp.abs


def velocity_effects(q, q_dot, mass_config, shape_config):
    du = jnp.zeros(7, dtype=jnp.float32)

    du[0] = (
        0.5
        * (
            mass_config[2]
            * (
                2
                * (
                    q_dot[3]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[0] * sin(q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[3]
                    * (
                        shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[5]
                    * (
                        shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    q_dot[5]
                    * (
                        -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
            )
            + mass_config[1]
            * (
                2
                * (
                    q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + -0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                )
                + 2
                * (
                    q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + -0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                )
                + 2
                * (
                    q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                )
                + 2
                * (
                    q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                )
            )
        )
        + -0.5
        * q_dot[0]
        * (
            mass_config[1]
            * (
                2
                * (
                    shape_config[0] * cos(q[0])
                    + 0.5 * shape_config[1] * cos(q[3] + q[0])
                )
                * (
                    q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + -0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                )
                + 2
                * (
                    q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                )
                * (
                    shape_config[0] * sin(q[0])
                    + 0.5 * shape_config[1] * sin(q[5] + q[0])
                )
                + 2
                * (
                    shape_config[0] * sin(q[0])
                    + 0.5 * shape_config[1] * sin(q[3] + q[0])
                )
                * (
                    q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                )
                + 2
                * (
                    -1 * shape_config[0] * sin(q[0])
                    + -0.5 * shape_config[1] * sin(q[3] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                )
                + 2
                * (
                    shape_config[0] * cos(q[0])
                    + 0.5 * shape_config[1] * cos(q[3] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                )
                + 2
                * (
                    shape_config[0] * cos(q[0])
                    + 0.5 * shape_config[1] * cos(q[5] + q[0])
                )
                * (
                    q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + -0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                )
                + 2
                * (
                    -1 * shape_config[0] * sin(q[0])
                    + -0.5 * shape_config[1] * sin(q[5] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                )
                + 2
                * (
                    shape_config[0] * cos(q[0])
                    + 0.5 * shape_config[1] * cos(q[5] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                )
            )
            + mass_config[2]
            * (
                2
                * (
                    -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    + -1 * shape_config[0] * sin(q[0])
                    + -1 * shape_config[1] * sin(q[3] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    -1 * shape_config[0] * sin(q[0])
                    + -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    + -1 * shape_config[1] * sin(q[5] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[5]
                    * (
                        -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[3]
                    * (
                        shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[5]
                    * (
                        shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    q_dot[3]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[0] * sin(q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
            )
        )
        + -0.5
        * q_dot[3]
        * (
            mass_config[2]
            * (
                2
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[3]
                    * (
                        shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[3]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    + -1 * shape_config[1] * sin(q[3] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
            )
            + mass_config[1]
            * (
                2
                * (
                    shape_config[0] * sin(q[0])
                    + 0.5 * shape_config[1] * sin(q[3] + q[0])
                )
                * (
                    0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                    + 0.5 * shape_config[1] * q_dot[0] * cos(q[3] + q[0])
                )
                + 2
                * (
                    shape_config[0] * cos(q[0])
                    + 0.5 * shape_config[1] * cos(q[3] + q[0])
                )
                * (
                    -0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                    + -0.5 * shape_config[1] * q_dot[0] * sin(q[3] + q[0])
                )
                + shape_config[1]
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                )
                * cos(q[3] + q[0])
                + -1.0
                * shape_config[1]
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                )
                * sin(q[3] + q[0])
            )
        )
        + -0.5
        * q_dot[5]
        * (
            mass_config[2]
            * (
                2
                * (
                    -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    + -1 * shape_config[1] * sin(q[5] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[5]
                    * (
                        shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[5]
                    * (
                        -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
            )
            + mass_config[1]
            * (
                2
                * (
                    shape_config[0] * cos(q[0])
                    + 0.5 * shape_config[1] * cos(q[5] + q[0])
                )
                * (
                    -0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                    + -0.5 * shape_config[1] * q_dot[0] * sin(q[5] + q[0])
                )
                + 2
                * (
                    shape_config[0] * sin(q[0])
                    + 0.5 * shape_config[1] * sin(q[5] + q[0])
                )
                * (
                    0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                    + 0.5 * shape_config[1] * q_dot[0] * cos(q[5] + q[0])
                )
                + shape_config[1]
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                )
                * cos(q[5] + q[0])
                + -1.0
                * shape_config[1]
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                )
                * sin(q[5] + q[0])
            )
        )
        + -0.5
        * mass_config[2]
        * q_dot[4]
        * (
            2
            * (
                shape_config[0] * sin(q[0])
                + shape_config[1] * sin(q[3] + q[0])
                + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
            )
            * (
                0.5 * shape_config[2] * q_dot[3] * cos(q[3] + q[4] + q[0])
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                + 0.5 * shape_config[2] * q_dot[0] * cos(q[3] + q[4] + q[0])
            )
            + 2
            * (
                shape_config[0] * cos(q[0])
                + shape_config[1] * cos(q[3] + q[0])
                + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
            )
            * (
                -0.5 * shape_config[2] * q_dot[3] * sin(q[3] + q[4] + q[0])
                + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                + -0.5 * shape_config[2] * q_dot[0] * sin(q[3] + q[4] + q[0])
            )
            + shape_config[2]
            * (
                q_dot[2]
                + q_dot[3]
                * (
                    shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
            )
            * cos(q[3] + q[4] + q[0])
            + -1.0
            * shape_config[2]
            * (
                q_dot[1]
                + q_dot[3]
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
            )
            * sin(q[3] + q[4] + q[0])
        )
        + -0.5
        * mass_config[2]
        * q_dot[6]
        * (
            2
            * (
                shape_config[0] * sin(q[0])
                + shape_config[1] * sin(q[5] + q[0])
                + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
            )
            * (
                0.5 * shape_config[2] * q_dot[5] * cos(q[5] + q[6] + q[0])
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                + 0.5 * shape_config[2] * q_dot[0] * cos(q[5] + q[6] + q[0])
            )
            + 2
            * (
                shape_config[0] * cos(q[0])
                + shape_config[1] * cos(q[5] + q[0])
                + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
            )
            * (
                -0.5 * shape_config[2] * q_dot[5] * sin(q[5] + q[6] + q[0])
                + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                + -0.5 * shape_config[2] * q_dot[0] * sin(q[5] + q[6] + q[0])
            )
            + shape_config[2]
            * (
                q_dot[2]
                + q_dot[5]
                * (
                    shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
            )
            * cos(q[5] + q[6] + q[0])
            + -1.0
            * shape_config[2]
            * (
                q_dot[1]
                + q_dot[5]
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
            )
            * sin(q[5] + q[6] + q[0])
        )
    )
    du[1] = (
        -0.5
        * q_dot[3]
        * (
            2
            * mass_config[1]
            * (
                -0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                + -0.5 * shape_config[1] * q_dot[0] * sin(q[3] + q[0])
            )
            + 2
            * mass_config[2]
            * (
                q_dot[3]
                * (
                    -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    + -1 * shape_config[1] * sin(q[3] + q[0])
                )
                + q_dot[0]
                * (
                    -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    + -1 * shape_config[1] * sin(q[3] + q[0])
                )
                + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
            )
        )
        + -0.5
        * q_dot[5]
        * (
            2
            * mass_config[1]
            * (
                -0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                + -0.5 * shape_config[1] * q_dot[0] * sin(q[5] + q[0])
            )
            + 2
            * mass_config[2]
            * (
                q_dot[5]
                * (
                    -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    + -1 * shape_config[1] * sin(q[5] + q[0])
                )
                + q_dot[0]
                * (
                    -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    + -1 * shape_config[1] * sin(q[5] + q[0])
                )
                + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
            )
        )
        + -0.5
        * q_dot[0]
        * (
            mass_config[2]
            * (
                2
                * (
                    q_dot[3]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[0] * sin(q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    q_dot[5]
                    * (
                        -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
            )
            + mass_config[1]
            * (
                2
                * (
                    q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + -0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                )
                + 2
                * (
                    q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + -0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                )
            )
        )
        + -1.0
        * mass_config[2]
        * q_dot[4]
        * (
            -0.5 * shape_config[2] * q_dot[3] * sin(q[3] + q[4] + q[0])
            + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
            + -0.5 * shape_config[2] * q_dot[0] * sin(q[3] + q[4] + q[0])
        )
        + -1.0
        * mass_config[2]
        * q_dot[6]
        * (
            -0.5 * shape_config[2] * q_dot[5] * sin(q[5] + q[6] + q[0])
            + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
            + -0.5 * shape_config[2] * q_dot[0] * sin(q[5] + q[6] + q[0])
        )
    )
    du[2] = (
        -0.5
        * q_dot[5]
        * (
            2
            * mass_config[2]
            * (
                q_dot[5]
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
            )
            + 2
            * mass_config[1]
            * (
                0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                + 0.5 * shape_config[1] * q_dot[0] * cos(q[5] + q[0])
            )
        )
        + -0.5
        * q_dot[3]
        * (
            2
            * mass_config[1]
            * (
                0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                + 0.5 * shape_config[1] * q_dot[0] * cos(q[3] + q[0])
            )
            + 2
            * mass_config[2]
            * (
                q_dot[3]
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
            )
        )
        + mass_config[2]
        * q_dot[4]
        * (
            -0.5 * shape_config[2] * q_dot[3] * cos(q[3] + q[4] + q[0])
            + -0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
            + -0.5 * shape_config[2] * q_dot[0] * cos(q[3] + q[4] + q[0])
        )
        + mass_config[2]
        * q_dot[6]
        * (
            -0.5 * shape_config[2] * q_dot[5] * cos(q[5] + q[6] + q[0])
            + -0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
            + -0.5 * shape_config[2] * q_dot[0] * cos(q[5] + q[6] + q[0])
        )
        + -0.5
        * q_dot[0]
        * (
            mass_config[2]
            * (
                2
                * (
                    q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
            )
            + mass_config[1]
            * (
                2
                * (
                    q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                )
                + 2
                * (
                    q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                )
            )
        )
    )
    du[3] = (
        0.5
        * (
            mass_config[2]
            * (
                2
                * (
                    q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[3]
                    * (
                        shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    q_dot[3]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
            )
            + mass_config[1]
            * (
                2
                * (
                    -0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                    + -0.5 * shape_config[1] * q_dot[0] * sin(q[3] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                )
                + 2
                * (
                    0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                    + 0.5 * shape_config[1] * q_dot[0] * cos(q[3] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                )
            )
        )
        + -0.5
        * q_dot[0]
        * (
            mass_config[2]
            * (
                2
                * (
                    shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[3]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[0] * sin(q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[3]
                    * (
                        shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    + -1 * shape_config[1] * sin(q[3] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
            )
            + mass_config[1]
            * (
                shape_config[1]
                * (
                    q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + -0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                )
                * cos(q[3] + q[0])
                + shape_config[1]
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                )
                * cos(q[3] + q[0])
                + shape_config[1]
                * (
                    q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                )
                * sin(q[3] + q[0])
                + -1
                * shape_config[1]
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                )
                * sin(q[3] + q[0])
            )
        )
        + -0.5
        * q_dot[3]
        * (
            mass_config[2]
            * (
                2
                * (
                    shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[3]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                        + -1 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[3]
                    * (
                        shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[3] + q[0])
                        + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                )
                + 2
                * (
                    -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    + -1 * shape_config[1] * sin(q[3] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[3]
                    * (
                        shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[3] + q[0])
                        + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                )
            )
            + mass_config[1]
            * (
                shape_config[1]
                * (
                    0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                    + 0.5 * shape_config[1] * q_dot[0] * cos(q[3] + q[0])
                )
                * sin(q[3] + q[0])
                + shape_config[1]
                * (
                    -0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                    + -0.5 * shape_config[1] * q_dot[0] * sin(q[3] + q[0])
                )
                * cos(q[3] + q[0])
                + shape_config[1]
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * sin(q[3] + q[0])
                )
                * cos(q[3] + q[0])
                + -1
                * shape_config[1]
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[3] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[3] * cos(q[3] + q[0])
                )
                * sin(q[3] + q[0])
            )
        )
        + -0.5
        * mass_config[2]
        * q_dot[4]
        * (
            2
            * (
                shape_config[1] * sin(q[3] + q[0])
                + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
            )
            * (
                0.5 * shape_config[2] * q_dot[3] * cos(q[3] + q[4] + q[0])
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                + 0.5 * shape_config[2] * q_dot[0] * cos(q[3] + q[4] + q[0])
            )
            + 2
            * (
                shape_config[1] * cos(q[3] + q[0])
                + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
            )
            * (
                -0.5 * shape_config[2] * q_dot[3] * sin(q[3] + q[4] + q[0])
                + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                + -0.5 * shape_config[2] * q_dot[0] * sin(q[3] + q[4] + q[0])
            )
            + shape_config[2]
            * (
                q_dot[2]
                + q_dot[3]
                * (
                    shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
            )
            * cos(q[3] + q[4] + q[0])
            + -1.0
            * shape_config[2]
            * (
                q_dot[1]
                + q_dot[3]
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
            )
            * sin(q[3] + q[4] + q[0])
        )
    )
    du[4] = (
        0.5
        * mass_config[2]
        * (
            2
            * (
                -0.5 * shape_config[2] * q_dot[3] * sin(q[3] + q[4] + q[0])
                + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                + -0.5 * shape_config[2] * q_dot[0] * sin(q[3] + q[4] + q[0])
            )
            * (
                q_dot[1]
                + q_dot[3]
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
            )
            + 2
            * (
                0.5 * shape_config[2] * q_dot[3] * cos(q[3] + q[4] + q[0])
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                + 0.5 * shape_config[2] * q_dot[0] * cos(q[3] + q[4] + q[0])
            )
            * (
                q_dot[2]
                + q_dot[3]
                * (
                    shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
            )
        )
        + -0.5
        * mass_config[2]
        * q_dot[3]
        * (
            shape_config[2]
            * (
                q_dot[3]
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
            )
            * sin(q[3] + q[4] + q[0])
            + shape_config[2]
            * (
                q_dot[3]
                * (
                    -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    + -1 * shape_config[1] * sin(q[3] + q[0])
                )
                + q_dot[0]
                * (
                    -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    + -1 * shape_config[1] * sin(q[3] + q[0])
                )
                + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
            )
            * cos(q[3] + q[4] + q[0])
            + shape_config[2]
            * (
                q_dot[2]
                + q_dot[3]
                * (
                    shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
            )
            * cos(q[3] + q[4] + q[0])
            + -1
            * shape_config[2]
            * (
                q_dot[1]
                + q_dot[3]
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
            )
            * sin(q[3] + q[4] + q[0])
        )
        + -0.5
        * mass_config[2]
        * q_dot[0]
        * (
            shape_config[2]
            * (
                q_dot[3]
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
            )
            * sin(q[3] + q[4] + q[0])
            + shape_config[2]
            * (
                q_dot[3]
                * (
                    -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    + -1 * shape_config[1] * sin(q[3] + q[0])
                )
                + q_dot[0]
                * (
                    -0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                    + -1 * shape_config[0] * sin(q[0])
                    + -1 * shape_config[1] * sin(q[3] + q[0])
                )
                + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
            )
            * cos(q[3] + q[4] + q[0])
            + shape_config[2]
            * (
                q_dot[2]
                + q_dot[3]
                * (
                    shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
            )
            * cos(q[3] + q[4] + q[0])
            + -1
            * shape_config[2]
            * (
                q_dot[1]
                + q_dot[3]
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
            )
            * sin(q[3] + q[4] + q[0])
        )
        + -0.5
        * mass_config[2]
        * q_dot[4]
        * (
            shape_config[2]
            * (
                -0.5 * shape_config[2] * q_dot[3] * sin(q[3] + q[4] + q[0])
                + -0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
                + -0.5 * shape_config[2] * q_dot[0] * sin(q[3] + q[4] + q[0])
            )
            * cos(q[3] + q[4] + q[0])
            + shape_config[2]
            * (
                q_dot[2]
                + q_dot[3]
                * (
                    shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[3] + q[0])
                    + 0.5 * shape_config[2] * sin(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * sin(q[3] + q[4] + q[0])
            )
            * cos(q[3] + q[4] + q[0])
            + shape_config[2]
            * (
                0.5 * shape_config[2] * q_dot[3] * cos(q[3] + q[4] + q[0])
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
                + 0.5 * shape_config[2] * q_dot[0] * cos(q[3] + q[4] + q[0])
            )
            * sin(q[3] + q[4] + q[0])
            + -1
            * shape_config[2]
            * (
                q_dot[1]
                + q_dot[3]
                * (
                    shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[3] + q[0])
                    + 0.5 * shape_config[2] * cos(q[3] + q[4] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[4] * cos(q[3] + q[4] + q[0])
            )
            * sin(q[3] + q[4] + q[0])
        )
    )
    du[5] = (
        0.5
        * (
            mass_config[2]
            * (
                2
                * (
                    q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[5]
                    * (
                        shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    q_dot[5]
                    * (
                        -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
            )
            + mass_config[1]
            * (
                2
                * (
                    0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                    + 0.5 * shape_config[1] * q_dot[0] * cos(q[5] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                )
                + 2
                * (
                    -0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                    + -0.5 * shape_config[1] * q_dot[0] * sin(q[5] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                )
            )
        )
        + -0.5
        * q_dot[0]
        * (
            mass_config[2]
            * (
                2
                * (
                    -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    + -1 * shape_config[1] * sin(q[5] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[5]
                    * (
                        -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[5]
                    * (
                        shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
            )
            + mass_config[1]
            * (
                shape_config[1]
                * (
                    q_dot[0]
                    * (
                        -1 * shape_config[0] * sin(q[0])
                        + -0.5 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + -0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                )
                * cos(q[5] + q[0])
                + shape_config[1]
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                )
                * cos(q[5] + q[0])
                + shape_config[1]
                * (
                    q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                )
                * sin(q[5] + q[0])
                + -1
                * shape_config[1]
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                )
                * sin(q[5] + q[0])
            )
        )
        + -0.5
        * q_dot[5]
        * (
            mass_config[2]
            * (
                2
                * (
                    -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    + -1 * shape_config[1] * sin(q[5] + q[0])
                )
                * (
                    q_dot[1]
                    + q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[5]
                    * (
                        -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + q_dot[0]
                    * (
                        -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                        + -1 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[2]
                    + q_dot[5]
                    * (
                        shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + shape_config[1] * sin(q[5] + q[0])
                        + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                )
                + 2
                * (
                    shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                * (
                    q_dot[5]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + q_dot[0]
                    * (
                        shape_config[1] * cos(q[5] + q[0])
                        + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                    )
                    + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                )
            )
            + mass_config[1]
            * (
                shape_config[1]
                * (
                    0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                    + 0.5 * shape_config[1] * q_dot[0] * cos(q[5] + q[0])
                )
                * sin(q[5] + q[0])
                + shape_config[1]
                * (
                    -0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                    + -0.5 * shape_config[1] * q_dot[0] * sin(q[5] + q[0])
                )
                * cos(q[5] + q[0])
                + shape_config[1]
                * (
                    q_dot[2]
                    + q_dot[0]
                    * (
                        shape_config[0] * sin(q[0])
                        + 0.5 * shape_config[1] * sin(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * sin(q[5] + q[0])
                )
                * cos(q[5] + q[0])
                + -1
                * shape_config[1]
                * (
                    q_dot[1]
                    + q_dot[0]
                    * (
                        shape_config[0] * cos(q[0])
                        + 0.5 * shape_config[1] * cos(q[5] + q[0])
                    )
                    + 0.5 * shape_config[1] * q_dot[5] * cos(q[5] + q[0])
                )
                * sin(q[5] + q[0])
            )
        )
        + -0.5
        * mass_config[2]
        * q_dot[6]
        * (
            2
            * (
                shape_config[1] * cos(q[5] + q[0])
                + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
            )
            * (
                -0.5 * shape_config[2] * q_dot[5] * sin(q[5] + q[6] + q[0])
                + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                + -0.5 * shape_config[2] * q_dot[0] * sin(q[5] + q[6] + q[0])
            )
            + 2
            * (
                shape_config[1] * sin(q[5] + q[0])
                + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
            )
            * (
                0.5 * shape_config[2] * q_dot[5] * cos(q[5] + q[6] + q[0])
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                + 0.5 * shape_config[2] * q_dot[0] * cos(q[5] + q[6] + q[0])
            )
            + shape_config[2]
            * (
                q_dot[2]
                + q_dot[5]
                * (
                    shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
            )
            * cos(q[5] + q[6] + q[0])
            + -1.0
            * shape_config[2]
            * (
                q_dot[1]
                + q_dot[5]
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
            )
            * sin(q[5] + q[6] + q[0])
        )
    )
    du[6] = (
        0.5
        * mass_config[2]
        * (
            2
            * (
                0.5 * shape_config[2] * q_dot[5] * cos(q[5] + q[6] + q[0])
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                + 0.5 * shape_config[2] * q_dot[0] * cos(q[5] + q[6] + q[0])
            )
            * (
                q_dot[2]
                + q_dot[5]
                * (
                    shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
            )
            + 2
            * (
                -0.5 * shape_config[2] * q_dot[5] * sin(q[5] + q[6] + q[0])
                + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                + -0.5 * shape_config[2] * q_dot[0] * sin(q[5] + q[6] + q[0])
            )
            * (
                q_dot[1]
                + q_dot[5]
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
            )
        )
        + -0.5
        * mass_config[2]
        * q_dot[5]
        * (
            shape_config[2]
            * (
                q_dot[5]
                * (
                    -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    + -1 * shape_config[1] * sin(q[5] + q[0])
                )
                + q_dot[0]
                * (
                    -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    + -1 * shape_config[1] * sin(q[5] + q[0])
                )
                + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
            )
            * cos(q[5] + q[6] + q[0])
            + shape_config[2]
            * (
                q_dot[2]
                + q_dot[5]
                * (
                    shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
            )
            * cos(q[5] + q[6] + q[0])
            + shape_config[2]
            * (
                q_dot[5]
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
            )
            * sin(q[5] + q[6] + q[0])
            + -1
            * shape_config[2]
            * (
                q_dot[1]
                + q_dot[5]
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
            )
            * sin(q[5] + q[6] + q[0])
        )
        + -0.5
        * mass_config[2]
        * q_dot[6]
        * (
            shape_config[2]
            * (
                -0.5 * shape_config[2] * q_dot[5] * sin(q[5] + q[6] + q[0])
                + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
                + -0.5 * shape_config[2] * q_dot[0] * sin(q[5] + q[6] + q[0])
            )
            * cos(q[5] + q[6] + q[0])
            + shape_config[2]
            * (
                q_dot[2]
                + q_dot[5]
                * (
                    shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
            )
            * cos(q[5] + q[6] + q[0])
            + shape_config[2]
            * (
                0.5 * shape_config[2] * q_dot[5] * cos(q[5] + q[6] + q[0])
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
                + 0.5 * shape_config[2] * q_dot[0] * cos(q[5] + q[6] + q[0])
            )
            * sin(q[5] + q[6] + q[0])
            + -1
            * shape_config[2]
            * (
                q_dot[1]
                + q_dot[5]
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
            )
            * sin(q[5] + q[6] + q[0])
        )
        + -0.5
        * mass_config[2]
        * q_dot[0]
        * (
            shape_config[2]
            * (
                q_dot[5]
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
            )
            * sin(q[5] + q[6] + q[0])
            + shape_config[2]
            * (
                q_dot[5]
                * (
                    -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    + -1 * shape_config[1] * sin(q[5] + q[0])
                )
                + q_dot[0]
                * (
                    -1 * shape_config[0] * sin(q[0])
                    + -0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                    + -1 * shape_config[1] * sin(q[5] + q[0])
                )
                + -0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
            )
            * cos(q[5] + q[6] + q[0])
            + shape_config[2]
            * (
                q_dot[2]
                + q_dot[5]
                * (
                    shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * sin(q[0])
                    + shape_config[1] * sin(q[5] + q[0])
                    + 0.5 * shape_config[2] * sin(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * sin(q[5] + q[6] + q[0])
            )
            * cos(q[5] + q[6] + q[0])
            + -1
            * shape_config[2]
            * (
                q_dot[1]
                + q_dot[5]
                * (
                    shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + q_dot[0]
                * (
                    shape_config[0] * cos(q[0])
                    + shape_config[1] * cos(q[5] + q[0])
                    + 0.5 * shape_config[2] * cos(q[5] + q[6] + q[0])
                )
                + 0.5 * shape_config[2] * q_dot[6] * cos(q[5] + q[6] + q[0])
            )
            * sin(q[5] + q[6] + q[0])
        )
    )

    return du
