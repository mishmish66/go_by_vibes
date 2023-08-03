from jax import numpy as jnp
from jax import grad, jit, vmap
from jax import random
from jax import lax


# class QuatRotation:
#     quat: jnp.ndarray

#     def __init__(self, quat: jnp.ndarray):
#         if quat.shape != (4):
#             raise ValueError("Quaternion must be 4")
#         self.quat = quat

#     @property
#     def w(self):
#         return self.quat[0]

#     @w.setter
#     def w(self, value):
#         self.quat[0] = value

#     @property
#     def x(self):
#         return self.quat[1]

#     @x.setter
#     def x(self, value):
#         self.quat[1] = value

#     @property
#     def y(self):
#         return self.quat[2]

#     @y.setter
#     def y(self, value):
#         self.quat[2] = value

#     @property
#     def z(self):
#         return self.quat[3]

#     @z.setter
#     def z(self, value):
#         self.quat[3] = value

#     @classmethod
#     def from_small_rotation(cls, small_rotation: "SmallRotation"):
#         angle = small_rotation.theta
#         half_angle = angle / 2
#         sin_half_angle, cos_half_angle = jnp.sin(half_angle), jnp.cos(half_angle)

#         w = cos_half_angle
#         x = sin_half_angle * small_rotation.x / angle
#         y = sin_half_angle * small_rotation.y / angle
#         z = sin_half_angle * small_rotation.z / angle

#         return cls(jnp.array([w, x, y, z]))

#     @classmethod
#     def from_big_rotation(cls, big_rotation: "BigRotation"):
#         result = cls(jnp.zeros(4))

#         trace = big_rotation.rot.trace()

#         def case_trace_greater_than_zero():
#             S = jnp.sqrt(trace + 1.0) * 2
#             result.w = 0.25 * S
#             result.x = (big_rotation.rot[2, 1] - big_rotation.rot[1, 2]) / S
#             result.y = (big_rotation.rot[0, 2] - big_rotation.rot[2, 0]) / S
#             result.z = (big_rotation.rot[1, 0] - big_rotation.rot[0, 1]) / S

#         def case_r11_greater_than_r22_and_r11_greater_than_r33():
#             S = (
#                 jnp.sqrt(
#                     1.0
#                     + big_rotation.rot[0, 0]
#                     - big_rotation.rot[1, 1]
#                     - big_rotation.rot[2, 2]
#                 )
#                 * 2
#             )
#             result.w = (big_rotation.rot[2, 1] - big_rotation.rot[1, 2]) / S
#             result.x = 0.25 * S
#             result.y = (big_rotation.rot[0, 1] + big_rotation.rot[1, 0]) / S
#             result.z = (big_rotation.rot[0, 2] + big_rotation.rot[2, 0]) / S

#         def case_r22_greater_than_r33():
#             S = (
#                 jnp.sqrt(
#                     1.0
#                     + big_rotation.rot[1, 1]
#                     - big_rotation.rot[0, 0]
#                     - big_rotation.rot[2, 2]
#                 )
#                 * 2
#             )
#             result.w = (big_rotation.rot[0, 2] - big_rotation.rot[2, 0]) / S
#             result.x = (big_rotation.rot[0, 1] + big_rotation.rot[1, 0]) / S
#             result.y = 0.25 * S
#             result.z = (big_rotation.rot[1, 2] + big_rotation.rot[2, 1]) / S

#         def case_r33_greater_than_r11_and_r33_greater_than_r22():
#             S = (
#                 jnp.sqrt(
#                     1.0
#                     + big_rotation.rot[2, 2]
#                     - big_rotation.rot[0, 0]
#                     - big_rotation.rot[1, 1]
#                 )
#                 * 2
#             )
#             result.w = (big_rotation.rot[1, 0] - big_rotation.rot[0, 1]) / S
#             result.x = (big_rotation.rot[0, 2] + big_rotation.rot[2, 0]) / S
#             result.y = (big_rotation.rot[1, 2] + big_rotation.rot[2, 1]) / S
#             result.z = 0.25 * S

#         def case_r11_less_than_r22_or_r11_less_than_r33():
#             r22_greater_than_r33 = big_rotation.rot[1, 1] > big_rotation.rot[2, 2]
#             lax.cond(
#                 r22_greater_than_r33,
#                 case_r22_greater_than_r33,
#                 case_r33_greater_than_r11_and_r33_greater_than_r22,
#             )

#         def case_trace_less_than_equal_to_zero():
#             r11_greater_than_r22 = big_rotation.rot[0, 0] > big_rotation.rot[1, 1]
#             r11_greater_than_r33 = big_rotation.rot[0, 0] > big_rotation.rot[2, 2]
#             lax.cond(
#                 r11_greater_than_r22 and r11_greater_than_r33,
#                 case_r11_greater_than_r22_and_r11_greater_than_r33,
#                 case_r11_less_than_r22_or_r11_less_than_r33,
#             )

#         lax.cond(
#             trace > 0, case_trace_greater_than_zero, case_trace_less_than_equal_to_zero
#         )

#         return cls(result)


# class SmallRotation:
#     rot: jnp.ndarray

#     def __init__(self, rot: jnp.ndarray):
#         if rot.shape != (3):
#             raise ValueError("Rotation vector must be 3")
#         self.rot = rot

#     @property
#     def x(self):
#         return self.rot[0]

#     @x.setter
#     def x(self, value):
#         self.rot[0] = value

#     @property
#     def y(self):
#         return self.rot[1]

#     @y.setter
#     def y(self, value):
#         self.rot[1] = value

#     @property
#     def z(self):
#         return self.rot[2]

#     @z.setter
#     def z(self, value):
#         self.rot[2] = value

#     @property
#     def theta(self):
#         return jnp.linalg.norm(self.rot)

#     @theta.setter
#     def theta(self, value):
#         if self.theta != 0:
#             self.rot = self.rot / self.theta * value

#     @classmethod
#     def from_quat(cls, quat: QuatRotation):
#         result = cls(jnp.zeros(3))

#         theta = jnp.arccos(quat.w) * 2
#         quat_axis_mag = jnp.sqrt(1 - quat.w**2)

#         ux = quat.x / quat_axis_mag
#         uy = quat.y / quat_axis_mag
#         uz = quat.z / quat_axis_mag

#         result.x = ux * theta
#         result.y = uy * theta
#         result.z = uz * theta

#     @classmethod
#     def from_big_rotation(cls, big_rotation: "BigRotation"):
#         result = cls(jnp.zeros(3))

#         trace = big_rotation.mat.trace()
#         cos_theta = (trace - 1) / 2
#         sin_theta = jnp.sqrt(1 - cos_theta**2)
#         theta = jnp.arccos(cos_theta)

#         if theta == 0:
#             return result

#         x = (big_rotation.mat[2, 1] - big_rotation.mat[1, 2]) / (2 * sin_theta)
#         y = (big_rotation.mat[0, 2] - big_rotation.mat[2, 0]) / (2 * sin_theta)
#         z = (big_rotation.mat[1, 0] - big_rotation.mat[0, 1]) / (2 * sin_theta)

#         result.x = x * theta
#         result.y = y * theta
#         result.z = z * theta

#         return result


# class BigRotation:
#     mat: jnp.ndarray

#     def __init__(self, mat: jnp.ndarray):
#         if mat.shape != (3, 3):
#             raise ValueError("Rotation matrix must be 3x3")
#         self.mat = mat


# class QuatTransform:
#     quat: jnp.ndarray
#     pos: jnp.ndarray

#     def __init__(self, quat: jnp.ndarray, pos: jnp.ndarray):
#         if quat.shape != (4):
#             raise ValueError("Quaternion must be 4")
#         if pos.shape != (3):
#             raise ValueError("Position vector must be 3")
#         self.quat = quat
#         self.pos = pos


def rotmat(theta):
    return jnp.array(
        [
            [jnp.cos(theta), -jnp.sin(theta)],
            [jnp.sin(theta), jnp.cos(theta)],
        ]
    )
