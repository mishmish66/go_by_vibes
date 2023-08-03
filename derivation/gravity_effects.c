#include <math.h>
void diffeqf(double* du, const double* RHS1, const double* RHS2, const double* RHS3, const double* RHS4, const double* RHS5, const double RHS6) {
  du[0] = RHS6 * (RHS4[1] * (2 * RHS5[0] * sin(RHS1[0]) + -0.5 * RHS5[1] * (-1 * sin(RHS1[3] + RHS1[0]) + -1 * sin(RHS1[5] + RHS1[0]))) + RHS4[2] * (RHS5[1] * (sin(RHS1[3] + RHS1[0]) + sin(RHS1[5] + RHS1[0])) + -0.5 * RHS5[2] * (-1 * sin(RHS1[3] + RHS1[4] + RHS1[0]) + -1 * sin(RHS1[5] + RHS1[6] + RHS1[0])) + 2 * RHS5[0] * sin(RHS1[0])));
  du[1] = 0;
  du[2] = RHS6 * RHS4[0] + RHS6 * (2 * RHS4[2] + 2 * RHS4[1]) + -1 * RHS4[2] * RHS2[4] * (-0.5 * RHS5[2] * RHS2[3] * cos(RHS1[3] + RHS1[4] + RHS1[0]) + -0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0]) + -0.5 * RHS5[2] * RHS2[0] * cos(RHS1[3] + RHS1[4] + RHS1[0])) + -1.0 * RHS4[2] * RHS2[4] * (0.5 * RHS5[2] * RHS2[3] * cos(RHS1[3] + RHS1[4] + RHS1[0]) + 0.5 * RHS5[2] * RHS2[4] * cos(RHS1[3] + RHS1[4] + RHS1[0]) + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[3] + RHS1[4] + RHS1[0])) + -1.0 * RHS4[2] * RHS2[6] * (0.5 * RHS5[2] * RHS2[5] * cos(RHS1[5] + RHS1[6] + RHS1[0]) + 0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0]) + 0.5 * RHS5[2] * RHS2[0] * cos(RHS1[5] + RHS1[6] + RHS1[0])) + -1 * RHS4[2] * RHS2[6] * (-0.5 * RHS5[2] * RHS2[5] * cos(RHS1[5] + RHS1[6] + RHS1[0]) + -0.5 * RHS5[2] * RHS2[6] * cos(RHS1[5] + RHS1[6] + RHS1[0]) + -0.5 * RHS5[2] * RHS2[0] * cos(RHS1[5] + RHS1[6] + RHS1[0]));
  du[3] = RHS6 * (RHS4[2] * (RHS5[1] * sin(RHS1[3] + RHS1[0]) + 0.5 * RHS5[2] * sin(RHS1[3] + RHS1[4] + RHS1[0])) + 0.5 * RHS5[1] * RHS4[1] * sin(RHS1[3] + RHS1[0]));
  du[4] = 0.5 * RHS6 * RHS5[2] * RHS4[2] * sin(RHS1[3] + RHS1[4] + RHS1[0]);
  du[5] = RHS6 * (RHS4[2] * (RHS5[1] * sin(RHS1[5] + RHS1[0]) + 0.5 * RHS5[2] * sin(RHS1[5] + RHS1[6] + RHS1[0])) + 0.5 * RHS5[1] * RHS4[1] * sin(RHS1[5] + RHS1[0]));
  du[6] = 0.5 * RHS6 * RHS5[2] * RHS4[2] * sin(RHS1[5] + RHS1[6] + RHS1[0]);
}
