import jax
import jax.numpy as jnp

pow = jnp.power
cos = jnp.cos
sin = jnp.sin
sqrt = jnp.sqrt
abs2 = jnp.abs


def mass_matrix(q, q_dot, mass_config, shape_config):
    result = diffeqf(q, None, None, mass_config, shape_config, None)

    result = result.reshape((7, 7))

    return result


def diffeqf(RHS1, RHS2, RHS3, RHS4, RHS5, RHS6):
    return jnp.array(
        [
            (
                -0.5
                * (
                    4 * RHS4[5]
                    + 4 * RHS4[4]
                    + RHS4[2]
                    * (
                        2
                        * pow(
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0]),
                            2,
                        )
                        + 2
                        * pow(
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0]),
                            2,
                        )
                        + 2
                        * pow(
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0]),
                            2,
                        )
                        + 2
                        * pow(
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0]),
                            2,
                        )
                    )
                    + RHS4[1]
                    * (
                        2
                        * pow(
                            RHS5[0] * cos(RHS1[0])
                            + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0]),
                            2,
                        )
                        + 2
                        * pow(
                            RHS5[0] * sin(RHS1[0])
                            + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0]),
                            2,
                        )
                        + 2
                        * pow(
                            RHS5[0] * cos(RHS1[0])
                            + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0]),
                            2,
                        )
                        + 2
                        * pow(
                            RHS5[0] * sin(RHS1[0])
                            + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0]),
                            2,
                        )
                    )
                )
                + -1.0 * RHS4[3]
            ),
            -0.5
            * (
                RHS4[1]
                * (
                    2
                    * (RHS5[0] * cos(RHS1[0]) + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0]))
                    + 2
                    * (RHS5[0] * cos(RHS1[0]) + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0]))
                )
                + RHS4[2]
                * (
                    2
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                )
            ),
            -0.5
            * (
                RHS4[1]
                * (
                    2
                    * (RHS5[0] * sin(RHS1[0]) + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0]))
                    + 2
                    * (RHS5[0] * sin(RHS1[0]) + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0]))
                )
                + RHS4[2]
                * (
                    2
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                )
            ),
            -0.5
            * (
                2 * RHS4[5]
                + 2 * RHS4[4]
                + RHS4[2]
                * (
                    2
                    * (
                        RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                )
                + RHS4[1]
                * (
                    RHS5[1]
                    * (RHS5[0] * cos(RHS1[0]) + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0]))
                    * cos(RHS1[3] + RHS1[0])
                    + RHS5[1]
                    * (RHS5[0] * sin(RHS1[0]) + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0]))
                    * sin(RHS1[3] + RHS1[0])
                )
            ),
            -0.5
            * (
                2 * RHS4[5]
                + RHS4[2]
                * (
                    RHS5[2]
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
            ),
            -0.5
            * (
                2 * RHS4[5]
                + 2 * RHS4[4]
                + RHS4[1]
                * (
                    RHS5[1]
                    * (RHS5[0] * cos(RHS1[0]) + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0]))
                    * cos(RHS1[5] + RHS1[0])
                    + RHS5[1]
                    * (RHS5[0] * sin(RHS1[0]) + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0]))
                    * sin(RHS1[5] + RHS1[0])
                )
                + RHS4[2]
                * (
                    2
                    * (
                        RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                )
            ),
            -0.5
            * (
                2 * RHS4[5]
                + RHS4[2]
                * (
                    RHS5[2]
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
            ),
            -0.5
            * (
                RHS4[1]
                * (
                    2
                    * (RHS5[0] * cos(RHS1[0]) + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0]))
                    + 2
                    * (RHS5[0] * cos(RHS1[0]) + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0]))
                )
                + RHS4[2]
                * (
                    2
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                )
            ),
            -0.5 * (4 * RHS4[2] + 4 * RHS4[1]) + -1.0 * RHS4[0],
            0,
            -0.5
            * (
                2
                * RHS4[2]
                * (
                    RHS5[1] * cos(RHS1[3] + RHS1[0])
                    + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                )
                + RHS5[1] * RHS4[1] * cos(RHS1[3] + RHS1[0])
            ),
            -0.5 * RHS5[2] * RHS4[2] * cos(RHS1[3] + RHS1[4] + RHS1[0]),
            -0.5
            * (
                2
                * RHS4[2]
                * (
                    RHS5[1] * cos(RHS1[5] + RHS1[0])
                    + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                )
                + RHS5[1] * RHS4[1] * cos(RHS1[5] + RHS1[0])
            ),
            -0.5 * RHS5[2] * RHS4[2] * cos(RHS1[5] + RHS1[6] + RHS1[0]),
            -0.5
            * (
                RHS4[1]
                * (
                    2
                    * (RHS5[0] * sin(RHS1[0]) + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0]))
                    + 2
                    * (RHS5[0] * sin(RHS1[0]) + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0]))
                )
                + RHS4[2]
                * (
                    2
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                )
            ),
            0,
            -0.5 * (4 * RHS4[2] + 4 * RHS4[1]) + -1.0 * RHS4[0],
            -0.5
            * (
                2
                * RHS4[2]
                * (
                    RHS5[1] * sin(RHS1[3] + RHS1[0])
                    + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
                + RHS5[1] * RHS4[1] * sin(RHS1[3] + RHS1[0])
            ),
            -0.5 * RHS5[2] * RHS4[2] * sin(RHS1[3] + RHS1[4] + RHS1[0]),
            -0.5
            * (
                2
                * RHS4[2]
                * (
                    RHS5[1] * sin(RHS1[5] + RHS1[0])
                    + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
                + RHS5[1] * RHS4[1] * sin(RHS1[5] + RHS1[0])
            ),
            -0.5 * RHS5[2] * RHS4[2] * sin(RHS1[5] + RHS1[6] + RHS1[0]),
            -0.5
            * (
                2 * RHS4[5]
                + 2 * RHS4[4]
                + RHS4[2]
                * (
                    2
                    * (
                        RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                )
                + RHS4[1]
                * (
                    RHS5[1]
                    * (RHS5[0] * cos(RHS1[0]) + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0]))
                    * cos(RHS1[3] + RHS1[0])
                    + RHS5[1]
                    * (RHS5[0] * sin(RHS1[0]) + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0]))
                    * sin(RHS1[3] + RHS1[0])
                )
            ),
            -0.5
            * (
                2
                * RHS4[2]
                * (
                    RHS5[1] * cos(RHS1[3] + RHS1[0])
                    + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                )
                + RHS5[1] * RHS4[1] * cos(RHS1[3] + RHS1[0])
            ),
            -0.5
            * (
                2
                * RHS4[2]
                * (
                    RHS5[1] * sin(RHS1[3] + RHS1[0])
                    + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
                + RHS5[1] * RHS4[1] * sin(RHS1[3] + RHS1[0])
            ),
            -0.5
            * (
                2 * RHS4[5]
                + 2 * RHS4[4]
                + RHS4[2]
                * (
                    2
                    * pow(
                        RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0]),
                        2,
                    )
                    + 2
                    * pow(
                        RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0]),
                        2,
                    )
                )
                + RHS4[1]
                * (
                    0.5 * pow(RHS5[1], 2) * pow(cos(RHS1[3] + RHS1[0]), 2)
                    + 0.5 * pow(RHS5[1], 2) * pow(sin(RHS1[3] + RHS1[0]), 2)
                )
            ),
            -0.5
            * (
                2 * RHS4[5]
                + RHS4[2]
                * (
                    RHS5[2]
                    * (
                        RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
            ),
            0,
            0,
            -0.5
            * (
                2 * RHS4[5]
                + RHS4[2]
                * (
                    RHS5[2]
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
            ),
            -0.5 * RHS5[2] * RHS4[2] * cos(RHS1[3] + RHS1[4] + RHS1[0]),
            -0.5 * RHS5[2] * RHS4[2] * sin(RHS1[3] + RHS1[4] + RHS1[0]),
            -0.5
            * (
                2 * RHS4[5]
                + RHS4[2]
                * (
                    RHS5[2]
                    * (
                        RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
            ),
            -0.5
            * (
                2 * RHS4[5]
                + RHS4[2]
                * (
                    0.5 * pow(RHS5[2], 2) * pow(cos(RHS1[3] + RHS1[4] + RHS1[0]), 2)
                    + 0.5 * pow(RHS5[2], 2) * pow(sin(RHS1[3] + RHS1[4] + RHS1[0]), 2)
                )
            ),
            0,
            0,
            -0.5
            * (
                2 * RHS4[5]
                + 2 * RHS4[4]
                + RHS4[1]
                * (
                    RHS5[1]
                    * (RHS5[0] * cos(RHS1[0]) + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0]))
                    * cos(RHS1[5] + RHS1[0])
                    + RHS5[1]
                    * (RHS5[0] * sin(RHS1[0]) + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0]))
                    * sin(RHS1[5] + RHS1[0])
                )
                + RHS4[2]
                * (
                    2
                    * (
                        RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                )
            ),
            -0.5
            * (
                2
                * RHS4[2]
                * (
                    RHS5[1] * cos(RHS1[5] + RHS1[0])
                    + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                )
                + RHS5[1] * RHS4[1] * cos(RHS1[5] + RHS1[0])
            ),
            -0.5
            * (
                2
                * RHS4[2]
                * (
                    RHS5[1] * sin(RHS1[5] + RHS1[0])
                    + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
                + RHS5[1] * RHS4[1] * sin(RHS1[5] + RHS1[0])
            ),
            0,
            0,
            -0.5
            * (
                2 * RHS4[5]
                + 2 * RHS4[4]
                + RHS4[2]
                * (
                    2
                    * pow(
                        RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0]),
                        2,
                    )
                    + 2
                    * pow(
                        RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0]),
                        2,
                    )
                )
                + RHS4[1]
                * (
                    0.5 * pow(RHS5[1], 2) * pow(cos(RHS1[5] + RHS1[0]), 2)
                    + 0.5 * pow(RHS5[1], 2) * pow(sin(RHS1[5] + RHS1[0]), 2)
                )
            ),
            -0.5
            * (
                2 * RHS4[5]
                + RHS4[2]
                * (
                    RHS5[2]
                    * (
                        RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
            ),
            -0.5
            * (
                2 * RHS4[5]
                + RHS4[2]
                * (
                    RHS5[2]
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
            ),
            -0.5 * RHS5[2] * RHS4[2] * cos(RHS1[5] + RHS1[6] + RHS1[0]),
            -0.5 * RHS5[2] * RHS4[2] * sin(RHS1[5] + RHS1[6] + RHS1[0]),
            0,
            0,
            -0.5
            * (
                2 * RHS4[5]
                + RHS4[2]
                * (
                    RHS5[2]
                    * (
                        RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
            ),
            -0.5
            * (
                2 * RHS4[5]
                + RHS4[2]
                * (
                    0.5 * pow(RHS5[2], 2) * pow(cos(RHS1[5] + RHS1[6] + RHS1[0]), 2)
                    + 0.5 * pow(RHS5[2], 2) * pow(sin(RHS1[5] + RHS1[6] + RHS1[0]), 2)
                )
            ),
        ],
        dtype=jnp.float32,
    )
