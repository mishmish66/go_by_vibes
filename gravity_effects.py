import jax
import jax.numpy as jnp

pow = jnp.power
cos = jnp.cos
sin = jnp.sin
sqrt = jnp.sqrt
abs2 = jnp.abs


def diffeqf(q, q_dot, mass_config, shape_config):
    du = jnp.zeros(49, dtype=jnp.float32)
#include <math.h>
void diffeqf(double* du, const double* q, const double* RHS2, const double* RHS3, const double* RHS4, const double* RHS5, const double RHS6) {
  du[0] = RHS6 * (RHS4[1] * (2 * RHS5[0] * sin(q[0]) + -0.5 * RHS5[1] * (-1 * sin(q[3] + q[0]) + -1 * sin(q[5] + q[0]))) + RHS4[2] * (RHS5[1] * (sin(q[3] + q[0]) + sin(q[5] + q[0])) + -0.5 * RHS5[2] * (-1 * sin(q[3] + q[4] + q[0]) + -1 * sin(q[5] + q[6] + q[0])) + 2 * RHS5[0] * sin(q[0])));
  du[1] = 0;
  du[2] = RHS6 * RHS4[0] + RHS6 * (2 * RHS4[2] + 2 * RHS4[1]) + -1 * RHS4[2] * RHS2[4] * (-0.5 * RHS5[2] * RHS2[3] * cos(q[3] + q[4] + q[0]) + -0.5 * RHS5[2] * RHS2[4] * cos(q[3] + q[4] + q[0]) + -0.5 * RHS5[2] * RHS2[0] * cos(q[3] + q[4] + q[0])) + -1.0 * RHS4[2] * RHS2[4] * (0.5 * RHS5[2] * RHS2[3] * cos(q[3] + q[4] + q[0]) + 0.5 * RHS5[2] * RHS2[4] * cos(q[3] + q[4] + q[0]) + 0.5 * RHS5[2] * RHS2[0] * cos(q[3] + q[4] + q[0])) + -1.0 * RHS4[2] * RHS2[6] * (0.5 * RHS5[2] * RHS2[5] * cos(q[5] + q[6] + q[0]) + 0.5 * RHS5[2] * RHS2[6] * cos(q[5] + q[6] + q[0]) + 0.5 * RHS5[2] * RHS2[0] * cos(q[5] + q[6] + q[0])) + -1 * RHS4[2] * RHS2[6] * (-0.5 * RHS5[2] * RHS2[5] * cos(q[5] + q[6] + q[0]) + -0.5 * RHS5[2] * RHS2[6] * cos(q[5] + q[6] + q[0]) + -0.5 * RHS5[2] * RHS2[0] * cos(q[5] + q[6] + q[0]));
  du[3] = RHS6 * (RHS4[2] * (RHS5[1] * sin(q[3] + q[0]) + 0.5 * RHS5[2] * sin(q[3] + q[4] + q[0])) + 0.5 * RHS5[1] * RHS4[1] * sin(q[3] + q[0]));
  du[4] = 0.5 * RHS6 * RHS5[2] * RHS4[2] * sin(q[3] + q[4] + q[0]);
  du[5] = RHS6 * (RHS4[2] * (RHS5[1] * sin(q[5] + q[0]) + 0.5 * RHS5[2] * sin(q[5] + q[6] + q[0])) + 0.5 * RHS5[1] * RHS4[1] * sin(q[5] + q[0]));
  du[6] = 0.5 * RHS6 * RHS5[2] * RHS4[2] * sin(q[5] + q[6] + q[0]);
}

    return du
