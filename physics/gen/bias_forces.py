import jax
import jax.numpy as jnp

pow = jnp.power
cos = jnp.cos
sin = jnp.sin
sqrt = jnp.sqrt
abs2 = jnp.abs


def bias_forces(q, qd, mass_config, shape_config, g):
    result = diffeqf(q, qd, None, mass_config, shape_config, g)

    return result


def diffeqf(RHS1, RHS2, RHS3, RHS4, RHS5, RHS6):
    return jnp.array(
        [
            (
                0.5
                * (
                    RHS4[2]
                    * (
                        2
                        * (
                            RHS2[3]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[0] * sin(RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[4]
                            * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[5]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[6]
                            * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[3]
                            * (
                                RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[5]
                            * (
                                RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                    )
                    + RHS4[1]
                    * (
                        2
                        * (
                            RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + -0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + -0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                        )
                    )
                )
                + RHS6
                * (
                    RHS4[1]
                    * (
                        2 * RHS5[0] * sin(RHS1[0])
                        + -0.5
                        * RHS5[1]
                        * (-1 * sin(RHS1[3] + RHS1[0]) + -1 * sin(RHS1[5] + RHS1[0]))
                    )
                    + RHS4[2]
                    * (
                        RHS5[1] * (sin(RHS1[3] + RHS1[0]) + sin(RHS1[5] + RHS1[0]))
                        + -0.5
                        * RHS5[2]
                        * (
                            -1 * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            + -1 * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2 * RHS5[0] * sin(RHS1[0])
                    )
                )
                + -0.5
                * RHS2[5]
                * (
                    RHS4[2]
                    * (
                        2
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[5]
                            * (
                                RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[5]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[6]
                            * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                    )
                    + RHS4[1]
                    * (
                        2
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        * (
                            0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[1] * RHS2[0] * cos(RHS1[5] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                        )
                        * (
                            -0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                            + -0.5 * RHS5[1] * RHS2[0] * sin(RHS1[5] + RHS1[0])
                        )
                        + RHS5[1]
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                        )
                        * cos(RHS1[5] + RHS1[0])
                        + -1.0
                        * RHS5[1]
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                        )
                        * sin(RHS1[5] + RHS1[0])
                    )
                )
                + -0.5
                * RHS2[3]
                * (
                    RHS4[1]
                    * (
                        2
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        * (
                            0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[1] * RHS2[0] * cos(RHS1[3] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                        )
                        * (
                            -0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                            + -0.5 * RHS5[1] * RHS2[0] * sin(RHS1[3] + RHS1[0])
                        )
                        + RHS5[1]
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                        )
                        * cos(RHS1[3] + RHS1[0])
                        + -1.0
                        * RHS5[1]
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                        )
                        * sin(RHS1[3] + RHS1[0])
                    )
                    + RHS4[2]
                    * (
                        2
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[3]
                            * (
                                RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[3]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[4]
                            * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                    )
                )
                + -0.5
                * RHS2[0]
                * (
                    RHS4[1]
                    * (
                        2
                        * (
                            RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + -0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                        )
                        + 2
                        * (
                            -1 * RHS5[0] * sin(RHS1[0])
                            + -0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + -0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                        )
                        + 2
                        * (
                            -1 * RHS5[0] * sin(RHS1[0])
                            + -0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                        )
                    )
                    + RHS4[2]
                    * (
                        2
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            + -1 * RHS5[0] * sin(RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            -1 * RHS5[0] * sin(RHS1[0])
                            + -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[3]
                            * (
                                RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[5]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[6]
                            * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[5]
                            * (
                                RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[3]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[0] * sin(RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[4]
                            * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                    )
                )
                + -0.5
                * RHS4[2]
                * RHS2[4]
                * (
                    2
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * (
                        0.5 * RHS5[2] * RHS2[3] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * (
                        -0.5 * RHS5[2] * RHS2[3] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[0] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + RHS5[2]
                    * (
                        RHS2[2]
                        + RHS2[3]
                        * (
                            RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + -1.0
                    * RHS5[2]
                    * (
                        RHS2[1]
                        + RHS2[3]
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
                + -0.5
                * RHS4[2]
                * RHS2[6]
                * (
                    2
                    * (
                        RHS5[0] * sin(RHS1[0])
                        + RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * (
                        0.5 * RHS5[2] * RHS2[5] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[0] * cos(RHS1[0])
                        + RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * (
                        -0.5 * RHS5[2] * RHS2[5] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[0] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    + RHS5[2]
                    * (
                        RHS2[2]
                        + RHS2[5]
                        * (
                            RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + -1.0
                    * RHS5[2]
                    * (
                        RHS2[1]
                        + RHS2[5]
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
            ),
            (
                -0.5
                * RHS2[3]
                * (
                    2
                    * RHS4[1]
                    * (
                        -0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                        + -0.5 * RHS5[1] * RHS2[0] * sin(RHS1[3] + RHS1[0])
                    )
                    + 2
                    * RHS4[2]
                    * (
                        RHS2[3]
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        + -0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                )
                + -0.5
                * RHS2[5]
                * (
                    2
                    * RHS4[1]
                    * (
                        -0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                        + -0.5 * RHS5[1] * RHS2[0] * sin(RHS1[5] + RHS1[0])
                    )
                    + 2
                    * RHS4[2]
                    * (
                        RHS2[5]
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        + -0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                )
                + -0.5
                * RHS2[0]
                * (
                    RHS4[2]
                    * (
                        2
                        * (
                            RHS2[3]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[0] * sin(RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[4]
                            * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[5]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[6]
                            * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                    )
                    + RHS4[1]
                    * (
                        2
                        * (
                            RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + -0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + -0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                        )
                    )
                )
                + -1.0
                * RHS4[2]
                * RHS2[4]
                * (
                    -0.5 * RHS5[2] * RHS2[3] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    + -0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    + -0.5 * RHS5[2] * RHS2[0] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
                + -1.0
                * RHS4[2]
                * RHS2[6]
                * (
                    -0.5 * RHS5[2] * RHS2[5] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    + -0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    + -0.5 * RHS5[2] * RHS2[0] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
            ),
            (
                RHS6 * RHS4[0]
                + RHS6 * (2 * RHS4[2] + 2 * RHS4[1])
                + -0.5
                * RHS2[3]
                * (
                    2
                    * RHS4[1]
                    * (
                        0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[1] * RHS2[0] * cos(RHS1[3] + RHS1[0])
                    )
                    + 2
                    * RHS4[2]
                    * (
                        RHS2[3]
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                )
                + -0.5
                * RHS2[0]
                * (
                    RHS4[2]
                    * (
                        2
                        * (
                            RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                    )
                    + RHS4[1]
                    * (
                        2
                        * (
                            RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                        )
                    )
                )
                + -0.5
                * RHS2[5]
                * (
                    2
                    * RHS4[2]
                    * (
                        RHS2[5]
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    + 2
                    * RHS4[1]
                    * (
                        0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[1] * RHS2[0] * cos(RHS1[5] + RHS1[0])
                    )
                )
                + -1.0
                * RHS4[2]
                * RHS2[4]
                * (
                    0.5 * RHS5[2] * RHS2[3] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                )
                + -1.0
                * RHS4[2]
                * RHS2[6]
                * (
                    0.5 * RHS5[2] * RHS2[5] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                )
            ),
            (
                RHS6
                * (
                    RHS4[2]
                    * (
                        RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + 0.5 * RHS5[1] * RHS4[1] * sin(RHS1[3] + RHS1[0])
                )
                + 0.5
                * (
                    RHS4[2]
                    * (
                        2
                        * (
                            RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[3]
                            * (
                                RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[3]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[4]
                            * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                    )
                    + RHS4[1]
                    * (
                        2
                        * (
                            0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[1] * RHS2[0] * cos(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                        )
                        + 2
                        * (
                            -0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                            + -0.5 * RHS5[1] * RHS2[0] * sin(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                        )
                    )
                )
                + -0.5
                * RHS2[0]
                * (
                    RHS4[2]
                    * (
                        2
                        * (
                            RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[3]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[0] * sin(RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[4]
                            * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[3]
                            * (
                                RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                    )
                    + RHS4[1]
                    * (
                        RHS5[1]
                        * (
                            RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + -0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                        )
                        * cos(RHS1[3] + RHS1[0])
                        + RHS5[1]
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                        )
                        * cos(RHS1[3] + RHS1[0])
                        + RHS5[1]
                        * (
                            RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                        )
                        * sin(RHS1[3] + RHS1[0])
                        + -1
                        * RHS5[1]
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                        )
                        * sin(RHS1[3] + RHS1[0])
                    )
                )
                + -0.5
                * RHS2[3]
                * (
                    RHS4[2]
                    * (
                        2
                        * (
                            RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[3]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[4]
                            * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[3]
                            * (
                                RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 2
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[3]
                            * (
                                RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[3] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                    )
                    + RHS4[1]
                    * (
                        RHS5[1]
                        * (
                            0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[1] * RHS2[0] * cos(RHS1[3] + RHS1[0])
                        )
                        * sin(RHS1[3] + RHS1[0])
                        + RHS5[1]
                        * (
                            -0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                            + -0.5 * RHS5[1] * RHS2[0] * sin(RHS1[3] + RHS1[0])
                        )
                        * cos(RHS1[3] + RHS1[0])
                        + RHS5[1]
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * sin(RHS1[3] + RHS1[0])
                        )
                        * cos(RHS1[3] + RHS1[0])
                        + -1
                        * RHS5[1]
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[3] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[3] * cos(RHS1[3] + RHS1[0])
                        )
                        * sin(RHS1[3] + RHS1[0])
                    )
                )
                + -0.5
                * RHS4[2]
                * RHS2[4]
                * (
                    2
                    * (
                        RHS5[1] * sin(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * (
                        0.5 * RHS5[2] * RHS2[3] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[1] * cos(RHS1[3] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * (
                        -0.5 * RHS5[2] * RHS2[3] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[0] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + RHS5[2]
                    * (
                        RHS2[2]
                        + RHS2[3]
                        * (
                            RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + -1.0
                    * RHS5[2]
                    * (
                        RHS2[1]
                        + RHS2[3]
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
            ),
            (
                0.5
                * RHS4[2]
                * (
                    2
                    * (
                        -0.5 * RHS5[2] * RHS2[3] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[0] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * (
                        RHS2[1]
                        + RHS2[3]
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    + 2
                    * (
                        0.5 * RHS5[2] * RHS2[3] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * (
                        RHS2[2]
                        + RHS2[3]
                        * (
                            RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                )
                + -0.5
                * RHS4[2]
                * RHS2[3]
                * (
                    RHS5[2]
                    * (
                        RHS2[3]
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS2[3]
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        + -0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS2[2]
                        + RHS2[3]
                        * (
                            RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + -1
                    * RHS5[2]
                    * (
                        RHS2[1]
                        + RHS2[3]
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
                + -0.5
                * RHS4[2]
                * RHS2[0]
                * (
                    RHS5[2]
                    * (
                        RHS2[3]
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS2[3]
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                            + -1 * RHS5[0] * sin(RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[3] + RHS1[0])
                        )
                        + -0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS2[2]
                        + RHS2[3]
                        * (
                            RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + -1
                    * RHS5[2]
                    * (
                        RHS2[1]
                        + RHS2[3]
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
                + -0.5
                * RHS4[2]
                * RHS2[4]
                * (
                    RHS5[2]
                    * (
                        -0.5 * RHS5[2] * RHS2[3] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[0] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS2[2]
                        + RHS2[3]
                        * (
                            RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    + RHS5[2]
                    * (
                        0.5 * RHS5[2] * RHS2[3] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                    + -1
                    * RHS5[2]
                    * (
                        RHS2[1]
                        + RHS2[3]
                        * (
                            RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[3] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0])
                    )
                    * sin(RHS1[3] + RHS1[4] + RHS1[0])
                )
                + 0.5 * RHS6 * RHS5[2] * RHS4[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])
            ),
            (
                0.5
                * (
                    RHS4[2]
                    * (
                        2
                        * (
                            RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[5]
                            * (
                                RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS2[5]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[6]
                            * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                    )
                    + RHS4[1]
                    * (
                        2
                        * (
                            0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[1] * RHS2[0] * cos(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                        )
                        + 2
                        * (
                            -0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                            + -0.5 * RHS5[1] * RHS2[0] * sin(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                        )
                    )
                )
                + RHS6
                * (
                    RHS4[2]
                    * (
                        RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    + 0.5 * RHS5[1] * RHS4[1] * sin(RHS1[5] + RHS1[0])
                )
                + -0.5
                * RHS2[0]
                * (
                    RHS4[2]
                    * (
                        2
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[5]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[6]
                            * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[5]
                            * (
                                RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                    )
                    + RHS4[1]
                    * (
                        RHS5[1]
                        * (
                            RHS2[0]
                            * (
                                -1 * RHS5[0] * sin(RHS1[0])
                                + -0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + -0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                        )
                        * cos(RHS1[5] + RHS1[0])
                        + RHS5[1]
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                        )
                        * cos(RHS1[5] + RHS1[0])
                        + RHS5[1]
                        * (
                            RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                        )
                        * sin(RHS1[5] + RHS1[0])
                        + -1
                        * RHS5[1]
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                        )
                        * sin(RHS1[5] + RHS1[0])
                    )
                )
                + -0.5
                * RHS2[5]
                * (
                    RHS4[2]
                    * (
                        2
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        * (
                            RHS2[1]
                            + RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[5]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                                + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + -0.5
                            * RHS5[2]
                            * RHS2[6]
                            * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[2]
                            + RHS2[5]
                            * (
                                RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + RHS5[1] * sin(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 2
                        * (
                            RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        * (
                            RHS2[5]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + RHS2[0]
                            * (
                                RHS5[1] * cos(RHS1[5] + RHS1[0])
                                + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                            )
                            + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                    )
                    + RHS4[1]
                    * (
                        RHS5[1]
                        * (
                            0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[1] * RHS2[0] * cos(RHS1[5] + RHS1[0])
                        )
                        * sin(RHS1[5] + RHS1[0])
                        + RHS5[1]
                        * (
                            -0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                            + -0.5 * RHS5[1] * RHS2[0] * sin(RHS1[5] + RHS1[0])
                        )
                        * cos(RHS1[5] + RHS1[0])
                        + RHS5[1]
                        * (
                            RHS2[2]
                            + RHS2[0]
                            * (
                                RHS5[0] * sin(RHS1[0])
                                + 0.5 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * sin(RHS1[5] + RHS1[0])
                        )
                        * cos(RHS1[5] + RHS1[0])
                        + -1
                        * RHS5[1]
                        * (
                            RHS2[1]
                            + RHS2[0]
                            * (
                                RHS5[0] * cos(RHS1[0])
                                + 0.5 * RHS5[1] * cos(RHS1[5] + RHS1[0])
                            )
                            + 0.5 * RHS5[1] * RHS2[5] * cos(RHS1[5] + RHS1[0])
                        )
                        * sin(RHS1[5] + RHS1[0])
                    )
                )
                + -0.5
                * RHS4[2]
                * RHS2[6]
                * (
                    2
                    * (
                        RHS5[1] * sin(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * (
                        0.5 * RHS5[2] * RHS2[5] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    + 2
                    * (
                        RHS5[1] * cos(RHS1[5] + RHS1[0])
                        + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * (
                        -0.5 * RHS5[2] * RHS2[5] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[0] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    + RHS5[2]
                    * (
                        RHS2[2]
                        + RHS2[5]
                        * (
                            RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + -1.0
                    * RHS5[2]
                    * (
                        RHS2[1]
                        + RHS2[5]
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
            ),
            (
                0.5
                * RHS4[2]
                * (
                    2
                    * (
                        0.5 * RHS5[2] * RHS2[5] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * (
                        RHS2[2]
                        + RHS2[5]
                        * (
                            RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    + 2
                    * (
                        -0.5 * RHS5[2] * RHS2[5] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[0] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * (
                        RHS2[1]
                        + RHS2[5]
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                )
                + -0.5
                * RHS4[2]
                * RHS2[5]
                * (
                    RHS5[2]
                    * (
                        RHS2[5]
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        + -0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS2[2]
                        + RHS2[5]
                        * (
                            RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS2[5]
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    + -1
                    * RHS5[2]
                    * (
                        RHS2[1]
                        + RHS2[5]
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
                + -0.5
                * RHS4[2]
                * RHS2[6]
                * (
                    RHS5[2]
                    * (
                        -0.5 * RHS5[2] * RHS2[5] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        + -0.5 * RHS5[2] * RHS2[0] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS2[2]
                        + RHS2[5]
                        * (
                            RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + RHS5[2]
                    * (
                        0.5 * RHS5[2] * RHS2[5] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    + -1
                    * RHS5[2]
                    * (
                        RHS2[1]
                        + RHS2[5]
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
                + -0.5
                * RHS4[2]
                * RHS2[0]
                * (
                    RHS5[2]
                    * (
                        RHS2[5]
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS2[5]
                        * (
                            -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            -1 * RHS5[0] * sin(RHS1[0])
                            + -0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                            + -1 * RHS5[1] * sin(RHS1[5] + RHS1[0])
                        )
                        + -0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + RHS5[2]
                    * (
                        RHS2[2]
                        + RHS2[5]
                        * (
                            RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * sin(RHS1[0])
                            + RHS5[1] * sin(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * sin(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    + -1
                    * RHS5[2]
                    * (
                        RHS2[1]
                        + RHS2[5]
                        * (
                            RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + RHS2[0]
                        * (
                            RHS5[0] * cos(RHS1[0])
                            + RHS5[1] * cos(RHS1[5] + RHS1[0])
                            + 0.5 * RHS5[2] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                        )
                        + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0])
                    )
                    * sin(RHS1[5] + RHS1[6] + RHS1[0])
                )
                + 0.5 * RHS6 * RHS5[2] * RHS4[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])
            ),
        ],
        dtype=jnp.float32,
    )
