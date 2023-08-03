# from jax import numpy as jnp
# from jax import grad, jit, vmap
# from jax import random
# from jax import lax

# from typing import List, OrderedDict


# class Rotation:
#     theta: jnp.float32

#     def __init__(self, theta: jnp.float32):
#         self.theta = theta


# class Transform:
#     theta: jnp.float32
#     x: jnp.float32
#     y: jnp.float32

#     def __init__(self, theta: jnp.float32, x_pos: jnp.float32, y_pos: jnp.float32):
#         self.theta = theta
#         self.x = x
#         self.y = y

#     def get_big_transform(self):
#         return jnp.array(
#             [
#                 [jnp.cos(self.theta), -jnp.sin(self.theta), self.x],
#                 [jnp.sin(self.theta), jnp.cos(self.theta), self.y],
#                 [0, 0, 1],
#             ]
#         )


# class Velocity:
#     theta: jnp.float32
#     x: jnp.float32
#     y: jnp.float32

#     def __init__(self, theta: jnp.float32, x_vel: jnp.float32, y_vel: jnp.float32):
#         self.theta = theta
#         self.x = x_vel
#         self.y = y_vel


# class ScrewAxis:
#     vec: jnp.ndarray

#     def __init__(self, vec: jnp.ndarray):
#         self.vec = vec

#     @property
#     def skew(self):
#         return jnp.array(
#             [
#                 [0, -self.vec[2], self.vec[1]],
#                 [self.vec[2], 0, -self.vec[0]],
#                 [-self.vec[1], self.vec[0], 0],
#             ]
#         )

#     def exp_map(self, theta: jnp.float32):


# class Link:
#     child_joints: List["Joint"]
#     POIs: OrderedDict[str, Transform]

#     def inv_newton_euler_outward_pass(
#         self, q: jnp.ndarray, q_dot: jnp.ndarray, q_ddot: jnp.ndarray
#     ):
#         q_i = q[0]
#         q_dot_i = q_dot[0]
#         q_ddot_i = q_ddot[0]


# class Joint:
#     ScrewAxis: jnp.ndarray
#     child_link: Link

#     def __init__(self, origin: Transform, child_link: Link):
#         self.origin = origin
#         self.child_link = child_link
